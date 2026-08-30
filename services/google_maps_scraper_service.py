from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import pandas as pd

from services.google_maps_url_utils import validate_google_maps_url


class GoogleMapsScraperService:
    """Collect visible Google Maps place reviews with local Playwright Chromium.

    The service deliberately avoids proxy rotation, CAPTCHA bypass and stealth plugins.
    It is intended for controlled project/demo use and converts collected records into
    the project's existing common review schema.
    """

    COMMON_COLUMNS = [
        "review_id",
        "domain",
        "entity_id",
        "entity_name",
        "review_text",
        "rating",
        "rating_original",
        "review_date",
        "source",
        "raw_source_path",
    ]

    SORT_LABELS = {
        "most_relevant": "Most relevant",
        "relevant": "Most relevant",
        "newest": "Newest",
        "highest": "Highest rating",
        "highest_rating": "Highest rating",
        "lowest": "Lowest rating",
        "lowest_rating": "Lowest rating",
    }

    # Google Maps uses several layouts. In newer layouts, reviews are opened
    # through a tab instead of the numeric review-count button. Keep the
    # semantic words separate from CSS classes so minor DOM changes are easier
    # to tolerate.
    # Strong words must identify an actual Reviews control. Generic words such
    # as "rating" are deliberately excluded because the Overview/Hours section
    # can contain rating text and was previously being mistaken for Reviews.
    REVIEW_WORDS = (
        "reviews", "review",
        "avis", "bewertungen", "rezensionen", "reseñas", "recensioni",
        "avaliações", "отзывы", "レビュー", "口コミ", "리뷰", "评论", "評論",
        "مراجعات", "समीक्षा", "रिव्यू",
    )

    NON_REVIEW_CONTROL_WORDS = (
        "hours", "open", "closed", "website", "directions", "save", "share",
        "call", "order", "menu", "overview", "photos", "about", "updates",
        "products", "services",
    )

    REVIEW_CARD_SELECTORS = (
        # Stable review-card identifier used by current Google Maps layouts.
        "div[data-review-id]",
        "div.jftiEf[data-review-id]",
        "div[class*='jftiEf'][data-review-id]",
        # Fallbacks for layouts where the ID is placed on a descendant/button.
        "div.jftiEf:has(span[jsname='bN97Pc'])",
        "div.jftiEf:has(span[jsname='fbQN7e'])",
        "div.jftiEf:has(div.MyEned span.wiI7pd)",
    )

    REVIEW_TEXT_SELECTORS = (
        # Current Google Maps review-text nodes (2026 layouts).
        "span[jsname='bN97Pc']",
        "span[jsname='fbQN7e']",
        # Older/current fallback layouts.
        "div.MyEned span.wiI7pd",
        ".MyEned span.wiI7pd",
        "span.wiI7pd",
        "[data-expandable-section] span",
    )

    def __init__(
        self,
        headless: Optional[bool] = None,
        locale: str = "en-US",
        timeout_ms: int = 45_000,
        slow_mo_ms: int = 0,
    ) -> None:
        if headless is None:
            raw = os.environ.get("GOOGLE_MAPS_HEADLESS", "true").strip().lower()
            headless = raw not in {"0", "false", "no", "off"}

        self.headless = bool(headless)
        self.locale = locale
        self.timeout_ms = int(timeout_ms)
        self.slow_mo_ms = max(0, int(slow_mo_ms))
        self.last_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # URL and metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def validate_place_url(value: str) -> str:
        return validate_google_maps_url(value)

    @staticmethod
    def _normalise_text(value: Any) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _safe_filename(value: str, fallback: str = "google_maps_place") -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_.")
        return (cleaned or fallback)[:100]

    @staticmethod
    def _rating_from_text(value: Any) -> Optional[float]:
        text = str(value or "")
        match = re.search(r"([1-5](?:[.,]\d+)?)\s*(?:star|stars)?", text, re.I)
        if not match:
            return None
        try:
            number = float(match.group(1).replace(",", "."))
            return max(1.0, min(5.0, number))
        except Exception:
            return None

    @staticmethod
    def _review_count_from_text(value: Any) -> Optional[int]:
        text = str(value or "")
        match = re.search(r"([\d,\.\s]+)\s+reviews?", text, re.I)
        if not match:
            return None
        digits = re.sub(r"\D", "", match.group(1))
        return int(digits) if digits else None

    @staticmethod
    def _entity_id_from_url(url: str) -> str:
        patterns = [
            r"!1s(0x[0-9a-fA-F]+:0x[0-9a-fA-F]+)",
            r"[?&]cid=(\d+)",
            r"[?&]ftid=([^&]+)",
            r"/place/([^/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)[:160]

        digest = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:24]
        return f"google_maps_{digest}"

    @staticmethod
    def _place_name_from_url(url: str) -> str:
        """Extract a readable place name from a standard Maps place URL."""
        match = re.search(r"/maps/place/([^/@?#]+)", str(url or ""), re.I)
        if not match:
            return ""
        value = unquote(match.group(1)).replace("+", " ")
        value = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", value)
        return GoogleMapsScraperService._normalise_text(value)

    @staticmethod
    def _coordinates_from_url(url: str) -> tuple[Optional[str], Optional[str]]:
        text = str(url or "")
        match = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", text)
        if not match:
            match = re.search(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)", text)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _search_navigation_url(self, place_url: str) -> str:
        """Build a search-based Maps URL for newer limited/direct layouts."""
        place_name = self._place_name_from_url(place_url)
        if not place_name:
            return ""
        lat, lng = self._coordinates_from_url(place_url)
        encoded = quote(place_name, safe="")
        if lat and lng:
            return f"https://www.google.com/maps/search/{encoded}/@{lat},{lng},17z?hl=en"
        return f"https://www.google.com/maps/search/{encoded}/?hl=en"

    @staticmethod
    def _reviews_url_from_current_url(current_url: str) -> str:
        """Create a conservative /reviews route as a final UI fallback."""
        url = str(current_url or "")
        if "/place/" not in url:
            return ""
        before, after = url.split("/place/", 1)
        place_segment = after.split("/", 1)[0].split("?", 1)[0]
        if not place_segment:
            return ""
        return f"{before}/place/{place_segment}/reviews?hl=en"

    @staticmethod
    def _infer_domain(category: str, entity_name: str, requested_domain: str) -> str:
        requested = str(requested_domain or "auto").strip().lower()
        if requested in {"hotel", "restaurant"}:
            return requested

        text = f"{category} {entity_name}".lower()
        hotel_terms = {
            "hotel", "resort", "motel", "hostel", "lodging", "guest house",
            "guesthouse", "inn", "bed and breakfast", "serviced apartment",
        }
        restaurant_terms = {
            "restaurant", "cafe", "café", "food", "diner", "bistro", "bakery",
            "pizza", "barbecue", "bbq", "buffet", "steakhouse", "fast food",
            "coffee shop", "tea house", "dhaba",
        }

        if any(term in text for term in hotel_terms):
            return "hotel"
        if any(term in text for term in restaurant_terms):
            return "restaurant"

        # The Maps input is currently intended for the project's hotel/restaurant flow.
        # Restaurant is the least surprising fallback, while the detected category is
        # kept in metadata so the user can override the domain in the UI.
        return "restaurant"

    # ------------------------------------------------------------------
    # Playwright locator helpers
    # ------------------------------------------------------------------
    def _first_visible(self, page, selectors: Iterable[str]):
        for selector in selectors:
            try:
                locator = page.locator(selector)
                count = min(locator.count(), 25)
                for index in range(count):
                    item = locator.nth(index)
                    if item.is_visible(timeout=500):
                        return item
            except Exception:
                continue
        return None

    @staticmethod
    def _locator_text(locator, selectors: Iterable[str], longest: bool = False) -> str:
        values: list[str] = []
        for selector in selectors:
            try:
                items = locator.locator(selector)
                for index in range(min(items.count(), 12)):
                    value = items.nth(index).inner_text(timeout=800)
                    value = GoogleMapsScraperService._normalise_text(value)
                    if value:
                        values.append(value)
            except Exception:
                continue

        if not values:
            return ""
        return max(values, key=len) if longest else values[0]

    @staticmethod
    def _locator_attribute(locator, selectors: Iterable[str], attribute: str) -> str:
        for selector in selectors:
            try:
                items = locator.locator(selector)
                for index in range(min(items.count(), 12)):
                    value = items.nth(index).get_attribute(attribute, timeout=800)
                    value = GoogleMapsScraperService._normalise_text(value)
                    if value:
                        return value
            except Exception:
                continue
        return ""

    def _dismiss_consent(self, page) -> None:
        labels = [
            "Accept all",
            "I agree",
            "Accept",
            "Agree",
            "Reject all",
            # Optional Google Maps feedback prompt. It can cover controls in the
            # lower-right corner and should never block place navigation.
            "No thanks",
        ]
        for label in labels:
            try:
                button = page.get_by_role("button", name=re.compile(rf"^{re.escape(label)}$", re.I)).first
                if button.is_visible(timeout=900):
                    button.click(timeout=2_500)
                    page.wait_for_timeout(700)
                    return
            except Exception:
                continue

    def _reviews_visible(self, page) -> bool:
        selectors = [
            "div[data-review-id]",
            "div.jftiEf[data-review-id]",
            "div[role='feed'] div[data-review-id]",
            "span[jsname='bN97Pc']",
            "span[jsname='fbQN7e']",
            "div.MyEned span.wiI7pd",
            "button[aria-label*='Sort reviews' i]",
            "button[aria-label='Sort' i]",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() > 0 and locator.first.is_visible(timeout=500):
                    return True
            except Exception:
                continue
        return False

    def _reviews_panel_selected(self, page) -> bool:
        """Detect the Reviews view before review cards have finished rendering.

        Google Maps can take several seconds to insert data-review-id cards. The
        old implementation treated that delay as a failed click and continued
        clicking other controls, which could open the Hours section.
        """
        try:
            current_url = str(page.url or "").lower()
            if "/reviews" in current_url:
                return True
        except Exception:
            pass

        selected_selectors = [
            "[role='tab'][aria-selected='true']",
            "[role='tab'][data-selected='true']",
            "[role='tab'].HHrUdb",
        ]
        for selector in selected_selectors:
            try:
                items = page.locator(selector)
                for index in range(min(items.count(), 20)):
                    item = items.nth(index)
                    aria = self._normalise_text(item.get_attribute("aria-label") or "").lower()
                    try:
                        label = self._normalise_text(item.inner_text(timeout=350) or "").lower()
                    except Exception:
                        label = ""
                    combined = f"{aria} {label}"
                    if any(word in combined for word in self.REVIEW_WORDS):
                        return True
            except Exception:
                continue

        # Review-only controls/chips can appear before the cards themselves.
        selectors = [
            "button[aria-label*='Sort reviews' i]",
            "button[aria-label='Sort' i]",
            "button:has-text('Write a review')",
            "div[role='main'] button:has-text('Newest')",
            "div[role='main'] button:has-text('Most relevant')",
        ]
        visible_hits = 0
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() > 0 and locator.first.is_visible(timeout=350):
                    visible_hits += 1
            except Exception:
                continue
        return visible_hits >= 1

    def _wait_for_review_content(self, page, timeout_ms: int = 12_000) -> bool:
        end_time = time.time() + (timeout_ms / 1000)
        while time.time() < end_time:
            if self._reviews_visible(page) or self._reviews_panel_selected(page):
                return True
            page.wait_for_timeout(250)
        return False

    def _element_review_score(self, item) -> float:
        """Score only controls that clearly represent Reviews."""
        try:
            aria = self._normalise_text(item.get_attribute("aria-label") or "").lower()
        except Exception:
            aria = ""
        try:
            text = self._normalise_text(item.inner_text(timeout=500) or "").lower()
        except Exception:
            text = ""
        try:
            role = self._normalise_text(item.get_attribute("role") or "").lower()
        except Exception:
            role = ""
        try:
            tab_index = self._normalise_text(item.get_attribute("data-tab-index") or "")
        except Exception:
            tab_index = ""

        combined = f"{aria} {text}".strip()
        strong_aria = any(word in aria for word in self.REVIEW_WORDS)
        strong_text = any(word in text for word in self.REVIEW_WORDS)
        numeric_reviews = bool(
            re.search(r"\d[\d,.\s]*\s+(?:reviews?|avis|bewertungen|rezensionen|reseñas|recensioni)", combined, re.I)
        )

        # Do not accept an element merely because it contains "rating". Hours
        # and Overview containers often contain the place's rating as well.
        if not (strong_aria or strong_text or numeric_reviews):
            return 0.0

        score = 0.0
        if strong_aria:
            score += 4.0
        if strong_text:
            score += 3.0
        if numeric_reviews:
            score += 4.0
        if role == "tab":
            score += 1.0
        if tab_index == "reviews":
            score += 1.0
        if any(term in combined for term in ("write a review", "add a review")):
            score -= 5.0
        if any(term in combined for term in self.NON_REVIEW_CONTROL_WORDS):
            score -= 4.0
        return score

    @staticmethod
    def _is_google_maps_page(page) -> bool:
        try:
            parsed = urlparse(str(page.url or ""))
            host = parsed.netloc.lower()
            path = parsed.path.lower()
            return ("google." in host or host.endswith(".google.com")) and "/maps" in path
        except Exception:
            return False

    def _click_review_candidate(self, page, item) -> bool:
        try:
            item.scroll_into_view_if_needed(timeout=2_000)
        except Exception:
            pass

        click_attempts = [
            lambda: item.click(timeout=4_000),
            lambda: item.click(timeout=4_000, force=True),
            lambda: item.evaluate("el => el.click()"),
            lambda: item.press("Enter", timeout=2_000),
        ]
        for click in click_attempts:
            try:
                click()
            except Exception:
                continue

            # A successful click must be allowed to settle. Do not click the
            # same candidate repeatedly while the Reviews panel is rendering.
            page.wait_for_timeout(900)

            # A review control must never navigate to an external website.
            # Recover the Maps tab immediately if a misleading control did so.
            if not self._is_google_maps_page(page):
                try:
                    page.go_back(wait_until="domcontentloaded", timeout=self.timeout_ms)
                    page.wait_for_timeout(1_000)
                except Exception:
                    pass
                continue

            return self._wait_for_review_content(page, timeout_ms=12_000)
        return False

    def _open_reviews_panel(self, page) -> None:
        if self._reviews_visible(page):
            return

        # Newer Maps layouts expose Reviews as a semantic tab. Check these
        # before looking for the older numeric review-count button.
        tab_selectors = [
            "[role='tab'][aria-label*='review' i]",
            "[role='tab']:has-text('Reviews')",
            "button[role='tab']",
            "div[role='tab']",
            "a[role='tab']",
            "[role='tab'][data-tab-index]",
        ]
        for selector in tab_selectors:
            try:
                items = page.locator(selector)
                candidates = []
                for index in range(min(items.count(), 40)):
                    item = items.nth(index)
                    score = self._element_review_score(item)
                    if score >= 3.0:
                        candidates.append((score, item))
                for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True):
                    if self._click_review_candidate(page, item):
                        return
            except Exception:
                continue

        direct_selectors = [
            "button[jsaction*='pane.reviewChart.moreReviews']",
            "button[jsaction*='pane.rating.moreReviews']",
            "div[jsaction*='pane.reviewChart.moreReviews']",
            "button[aria-label*='review' i]",
            "[role='button'][aria-label*='review' i]",
            "[role='tab'][aria-label*='review' i]",
            "button:has-text('reviews')",
            "div[role='button']:has-text('reviews')",
        ]
        for selector in direct_selectors:
            try:
                items = page.locator(selector)
                candidates = []
                for index in range(min(items.count(), 80)):
                    item = items.nth(index)
                    score = self._element_review_score(item)
                    if score >= 3.0:
                        candidates.append((score, item))
                for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True):
                    # Text can be inside a span. Prefer the closest clickable parent.
                    clickable = item
                    try:
                        parent = item.locator(
                            "xpath=ancestor-or-self::button | "
                            "ancestor-or-self::*[@role='button'] | "
                            "ancestor-or-self::*[@role='tab']"
                        ).last
                        if parent.count() > 0:
                            clickable = parent
                    except Exception:
                        pass
                    if self._click_review_candidate(page, clickable):
                        return
            except Exception:
                continue

        # Last UI fallback: inspect visible semantic elements in the DOM and
        # click the highest-scoring clickable ancestor. This avoids depending
        # on Google's generated class names.
        try:
            clicked = page.evaluate("""
                () => {
                    const words = ['review', 'reviews'];
                    const blocked = ['hours', 'open', 'closed', 'website', 'directions', 'save', 'share', 'menu', 'overview', 'photos', 'about'];
                    const nodes = Array.from(document.querySelectorAll(
                        '[role=tab], button, [role=button], [aria-label]'
                    ));
                    const candidates = nodes.map(el => {
                        const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                        const text = (el.innerText || el.textContent || '').trim().toLowerCase();
                        const combined = `${aria} ${text}`;
                        let score = 0;
                        if (words.some(w => aria.includes(w))) score += 4;
                        if (words.some(w => text.includes(w))) score += 3;
                        if (/[0-9][0-9,. ]* +reviews?/i.test(combined)) score += 4;
                        if ((el.getAttribute('role') || '') === 'tab') score += 1;
                        if (combined.includes('write a review')) score -= 5;
                        if (blocked.some(w => combined.includes(w))) score -= 4;
                        return {el, score};
                    }).filter(x => x.score >= 3)
                      .sort((a, b) => b.score - a.score);
                    for (const candidate of candidates) {
                        const el = candidate.el.closest('button,[role=button],[role=tab]') || candidate.el;
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            el.scrollIntoView({block: 'center'});
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            if clicked:
                page.wait_for_timeout(1_000)
                if self._wait_for_review_content(page, timeout_ms=6_000):
                    return
        except Exception:
            pass

        # Google also supports a place /reviews route in some layouts. It is a
        # final fallback only; the normal UI click remains the primary method.
        reviews_url = self._reviews_url_from_current_url(page.url)
        if reviews_url:
            try:
                page.goto(reviews_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                page.wait_for_timeout(2_000)
                self._dismiss_consent(page)
                if self._wait_for_review_content(page, timeout_ms=8_000):
                    return
            except Exception:
                pass

        raise RuntimeError(
            "Could not open the Google Maps reviews panel. The current Maps layout did not "
            "expose a usable Reviews tab/button. Run the test with --headed and check the "
            "debug files under outputs/google_maps_debug."
        )

    def _apply_sort(self, page, sort_order: str) -> str:
        key = str(sort_order or "most_relevant").strip().lower()
        label = self.SORT_LABELS.get(key, "Most relevant")

        if label == "Most relevant":
            return "most_relevant"

        sort_button = self._first_visible(
            page,
            [
                "button[aria-label*='Sort reviews']",
                "button[aria-label^='Sort']",
                "button[data-value='Sort']",
            ],
        )
        if sort_button is None:
            try:
                sort_button = page.get_by_role("button", name=re.compile(r"^Sort$|Sort reviews", re.I)).first
            except Exception:
                sort_button = None

        if sort_button is None:
            return "most_relevant"

        try:
            sort_button.click(timeout=4_000)
            page.wait_for_timeout(500)
        except Exception:
            return "most_relevant"

        menu_candidates = [
            page.get_by_role("menuitemradio", name=re.compile(rf"^{re.escape(label)}$", re.I)),
            page.get_by_role("menuitem", name=re.compile(rf"^{re.escape(label)}$", re.I)),
            page.get_by_text(re.compile(rf"^{re.escape(label)}$", re.I)),
        ]
        for candidate in menu_candidates:
            try:
                item = candidate.first
                if item.is_visible(timeout=1_500):
                    item.click(timeout=4_000)
                    page.wait_for_timeout(1_300)
                    return key
            except Exception:
                continue

        # Close an open menu if the selected option was not found.
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
        return "most_relevant"

    def _find_feed(self, page):
        """Return the scroll container that actually contains review cards.

        Google Maps may expose multiple role=feed elements (search results, photos,
        suggestions). Selecting the first visible feed can therefore point at the wrong
        pane. Prefer a container that contains data-review-id cards or review-text nodes.
        """
        selectors = [
            "div[role='feed']",
            "div[aria-label*='Reviews' i]",
            "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",
            "div[role='main'] div.m6QErb",
        ]
        best = None
        best_score = -1
        for selector in selectors:
            try:
                items = page.locator(selector)
                for index in range(min(items.count(), 30)):
                    item = items.nth(index)
                    if not item.is_visible(timeout=400):
                        continue
                    review_cards = item.locator("div[data-review-id]").count()
                    review_texts = item.locator(
                        "span[jsname='bN97Pc'], span[jsname='fbQN7e'], "
                        "div.MyEned span.wiI7pd"
                    ).count()
                    try:
                        aria = self._normalise_text(item.get_attribute("aria-label") or "").lower()
                    except Exception:
                        aria = ""
                    score = (review_cards * 10) + (review_texts * 3)
                    if "review" in aria:
                        score += 5
                    if score > best_score:
                        best = item
                        best_score = score
            except Exception:
                continue
        return best if best_score > 0 else None

    def _review_cards(self, page, feed=None):
        # First use the stable global data-review-id selector. This avoids accidentally
        # returning unrelated jftiEf search-result cards from another visible feed.
        try:
            cards = page.locator("div[data-review-id]")
            if cards.count() > 0:
                return cards
        except Exception:
            pass

        root = feed if feed is not None else page
        for selector in self.REVIEW_CARD_SELECTORS[1:]:
            try:
                cards = root.locator(selector)
                if cards.count() > 0:
                    return cards
            except Exception:
                continue

        return page.locator(
            "div.jftiEf:has(span[jsname='bN97Pc']), "
            "div.jftiEf:has(span[jsname='fbQN7e']), "
            "div.jftiEf:has(div.MyEned span.wiI7pd)"
        )

    def _expand_review_card(self, card) -> None:
        """Expand only the current review body's truncated text.

        Never search for a generic ``More`` button at page/feed level. Google Maps
        uses the same word for Hours, prices and external booking actions; clicking
        those controls caused the scraper to leave Reviews or navigate to a hotel site.
        """
        selectors = [
            "button.w8nwRe.kyuRq",
            "button[jsaction*='expandReview' i]",
            "button[aria-label='More'].w8nwRe",
            "button[aria-label='More'][jsaction*='review' i]",
        ]
        for selector in selectors:
            try:
                buttons = card.locator(selector)
                for index in range(min(buttons.count(), 4)):
                    button = buttons.nth(index)
                    try:
                        if not button.is_visible(timeout=250):
                            continue
                        label = self._normalise_text(button.inner_text(timeout=250) or "").lower()
                        aria = self._normalise_text(button.get_attribute("aria-label") or "").lower()
                        combined = f"{label} {aria}".strip()

                        # The class/jsaction selectors are already review-specific. For
                        # aria-only fallbacks, require an exact review expansion label.
                        if "aria-label" in selector and combined not in {
                            "more", "read more", "more more", "read more read more"
                        }:
                            continue

                        button.click(timeout=900)
                        return
                    except Exception:
                        continue
            except Exception:
                continue

    def _fallback_review_text(
        self,
        card,
        reviewer_name: str = "",
        review_date: str = "",
        owner_response: str = "",
    ) -> str:
        """Select the most likely review body when generated CSS classes change."""
        try:
            values = card.evaluate(
                r"""
                (el) => {
                    const selectors = [
                        'span[jsname="bN97Pc"]',
                        'span[jsname="fbQN7e"]',
                        'div.MyEned span.wiI7pd',
                        'span.wiI7pd',
                        '[data-expandable-section] span',
                        'span[lang]',
                        'div[lang]'
                    ];
                    const out = [];
                    for (const selector of selectors) {
                        for (const node of el.querySelectorAll(selector)) {
                            const text = (node.innerText || node.textContent || '')
                                .replace(/\s+/g, ' ').trim();
                            if (text) out.push(text);
                        }
                    }
                    return [...new Set(out)];
                }
                """
            )
        except Exception:
            values = []

        ignore_exact = {
            self._normalise_text(reviewer_name).lower(),
            self._normalise_text(review_date).lower(),
            self._normalise_text(owner_response).lower(),
            "more", "less", "like", "share",
        }
        candidates: list[str] = []
        for value in values or []:
            text = self._normalise_text(value)
            low = text.lower()
            if not text or low in ignore_exact:
                continue
            if len(text) < 3:
                continue
            if re.fullmatch(r"[1-5](?:[.,]\d+)?(?:\s*stars?)?", low):
                continue
            if re.fullmatch(r"(?:a|an|one|\d+)\s+(?:day|week|month|year)s?\s+ago", low):
                continue
            if low.startswith("response from the owner"):
                continue
            if "local guide" in low and len(text) < 80:
                continue
            candidates.append(text)

        return max(candidates, key=len) if candidates else ""

    def _extract_card(self, card, entity: Dict[str, Any], source_url: str) -> Optional[Dict[str, Any]]:
        # Expand text only inside this verified review card. This cannot click
        # Hours/More prices/external booking controls elsewhere on the page.
        self._expand_review_card(card)

        try:
            review_id = self._normalise_text(card.get_attribute("data-review-id") or "")
        except Exception:
            review_id = ""

        reviewer_name = self._locator_text(
            card,
            [
                ".d4r55",
                "div[class*='d4r55']",
                "button.WNxzHc",
                ".WNxzHc",
                "a[href*='/maps/contrib/']",
            ],
        )

        rating_label = self._locator_attribute(
            card,
            [
                "span[role='img'][aria-label]",
                "span.kvMYJc",
                "span[class*='kvMYJc']",
                "span[aria-label*='star' i]",
            ],
            "aria-label",
        )
        rating = self._rating_from_text(rating_label)

        review_date = self._locator_text(
            card,
            [
                "span[class*='rsqaWe']",
                "span[class*='xRkPPb']",
                ".rsqaWe",
                ".DU9Pgb",
            ],
        )

        # Owner-response text is stored in raw output but excluded from review_text.
        owner_response = self._locator_text(
            card,
            [
                ".CDe7pd span.wiI7pd",
                ".CDe7pd div.wiI7pd",
                ".CDe7pd .wiI7pd",
                "div[aria-label*='Response from the owner' i] span",
            ],
            longest=True,
        )

        review_text = self._locator_text(card, self.REVIEW_TEXT_SELECTORS, longest=True)
        review_text = self._normalise_text(review_text)

        if not review_text:
            review_text = self._fallback_review_text(
                card,
                reviewer_name=reviewer_name,
                review_date=review_date,
                owner_response=owner_response,
            )

        review_text = self._normalise_text(review_text)
        if not review_text:
            return None

        if not review_id:
            fingerprint = "|".join(
                [entity["entity_id"], reviewer_name, str(rating or ""), review_date, review_text]
            )
            review_id = hashlib.sha256(fingerprint.encode("utf-8", errors="ignore")).hexdigest()

        return {
            "platform": "google_maps",
            "domain": entity["domain"],
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "category": entity.get("category", ""),
            "overall_rating": entity.get("overall_rating"),
            "displayed_review_count": entity.get("displayed_review_count"),
            "review_id": review_id,
            "reviewer_name": reviewer_name,
            "rating": rating,
            "review_text": review_text,
            "review_date": review_date,
            "owner_response": owner_response,
            "source_url": source_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _extract_place_metadata(self, page, requested_domain: str) -> Dict[str, Any]:
        name_locator = self._first_visible(page, ["h1.DUwDvf", "h1", "[role='main'] h1"])
        entity_name = "Google Maps Place"
        if name_locator is not None:
            try:
                entity_name = self._normalise_text(name_locator.inner_text(timeout=1_500)) or entity_name
            except Exception:
                pass

        category_locator = self._first_visible(
            page,
            [
                "button.DkEaL",
                "button[jsaction*='pane.rating.category']",
                "button[aria-label*='Category']",
            ],
        )
        category = ""
        if category_locator is not None:
            try:
                category = self._normalise_text(category_locator.inner_text(timeout=1_000))
            except Exception:
                pass

        overall_rating = None
        for selector in ["div.F7nice span[aria-hidden='true']", "span.ceNzKf", "div.fontDisplayLarge"]:
            try:
                values = page.locator(selector)
                for index in range(min(values.count(), 8)):
                    text = self._normalise_text(values.nth(index).inner_text(timeout=500))
                    match = re.fullmatch(r"[1-5](?:[.,]\d+)?", text)
                    if match:
                        overall_rating = float(text.replace(",", "."))
                        break
                if overall_rating is not None:
                    break
            except Exception:
                continue

        displayed_review_count = None
        try:
            candidates = page.locator("button[aria-label*='review'], span[aria-label*='review']")
            for index in range(min(candidates.count(), 40)):
                text = " ".join(
                    [
                        self._normalise_text(candidates.nth(index).get_attribute("aria-label") or ""),
                        self._normalise_text(candidates.nth(index).inner_text(timeout=400) or ""),
                    ]
                )
                displayed_review_count = self._review_count_from_text(text)
                if displayed_review_count is not None:
                    break
        except Exception:
            pass

        final_url = page.url
        domain = self._infer_domain(category, entity_name, requested_domain)
        return {
            "entity_id": self._entity_id_from_url(final_url),
            "entity_name": entity_name,
            "category": category,
            "domain": domain,
            "overall_rating": overall_rating,
            "displayed_review_count": displayed_review_count,
            "resolved_url": final_url,
        }

    def _save_debug_artifacts(self, page, reason: str = "failure") -> Dict[str, str]:
        debug_dir = Path(os.environ.get("GOOGLE_MAPS_DEBUG_DIR", "outputs/google_maps_debug"))
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_reason = self._safe_filename(reason, fallback="failure")
        screenshot_path = debug_dir / f"maps_{safe_reason}_{stamp}.png"
        html_path = debug_dir / f"maps_{safe_reason}_{stamp}.html"
        info_path = debug_dir / f"maps_{safe_reason}_{stamp}.txt"
        paths: Dict[str, str] = {}
        try:
            page.screenshot(path=str(screenshot_path), full_page=False)
            paths["screenshot"] = str(screenshot_path)
        except Exception:
            pass
        try:
            html_path.write_text(page.content(), encoding="utf-8")
            paths["html"] = str(html_path)
        except Exception:
            pass
        try:
            lines = [f"URL: {page.url}", f"Title: {page.title()}"]
            candidates = page.locator("[role='tab'], button, [role='button'], [aria-label]")
            for index in range(min(candidates.count(), 150)):
                item = candidates.nth(index)
                score = self._element_review_score(item)
                if score <= 0:
                    continue
                try:
                    lines.append(
                        " | ".join(
                            [
                                f"score={score}",
                                f"tag={item.evaluate('el => el.tagName')}",
                                f"role={item.get_attribute('role') or ''}",
                                f"aria={self._normalise_text(item.get_attribute('aria-label') or '')}",
                                f"text={self._normalise_text(item.inner_text(timeout=300) or '')[:250]}",
                            ]
                        )
                    )
                except Exception:
                    continue
            info_path.write_text("\n".join(lines), encoding="utf-8")
            paths["info"] = str(info_path)
        except Exception:
            pass
        return paths

    @staticmethod
    def _name_match_score(left: str, right: str) -> float:
        """Return a conservative similarity score for two visible place names.

        Google can append locality/category text or vary punctuation. Token overlap
        is therefore safer than strict equality, while still preventing a generic
        search-results page from being accepted as the requested place.
        """
        def normalise(value: str) -> str:
            value = GoogleMapsScraperService._normalise_text(value).lower()
            value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
            return re.sub(r"\s+", " ", value).strip()

        a = normalise(left)
        b = normalise(right)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            shorter = min(len(a), len(b))
            longer = max(len(a), len(b))
            return max(0.82, shorter / max(1, longer))

        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = len(a_tokens & b_tokens)
        precision = intersection / len(a_tokens)
        recall = intersection / len(b_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _place_pane_matches(self, page, requested_name: str) -> bool:
        """Verify that Maps has opened one concrete place, not a result list."""
        headings = []
        for selector in ["h1.DUwDvf", "[role='main'] h1", "h1"]:
            try:
                items = page.locator(selector)
                for index in range(min(items.count(), 8)):
                    item = items.nth(index)
                    if not item.is_visible(timeout=350):
                        continue
                    text = self._normalise_text(item.inner_text(timeout=500) or "")
                    if text and text not in headings:
                        headings.append(text)
            except Exception:
                continue

        if not headings:
            return False

        # A selected place normally lives on /maps/place/. Search-result pages can
        # contain headings too, so a matching title is mandatory when we know it.
        requested = self._normalise_text(requested_name)
        if requested:
            if max((self._name_match_score(requested, h) for h in headings), default=0.0) < 0.80:
                return False

        try:
            current_url = str(page.url or "").lower()
            if "/maps/place/" in current_url or "/place/" in current_url:
                return True
        except Exception:
            pass

        # Some localized Maps URLs retain /search/ after a result is selected.
        # Require place-level tabs/actions as a second strong signal.
        place_signals = [
            "button:has-text('Directions')",
            "button:has-text('Save')",
            "[role='tab']:has-text('Reviews')",
            "button:has-text('Reviews')",
            "[aria-label*='Directions' i]",
        ]
        hits = 0
        for selector in place_signals:
            try:
                loc = page.locator(selector)
                if loc.count() > 0 and loc.first.is_visible(timeout=350):
                    hits += 1
            except Exception:
                continue
        return hits >= 2

    def _open_matching_search_result(self, page, requested_name: str) -> bool:
        """Select the requested result when Google redirects to a search list."""
        if not requested_name:
            return False

        selectors = [
            "div[role='feed'] a[href*='/maps/place/']",
            "a.hfpxzc[href*='/maps/place/']",
            "a[href*='/maps/place/'][aria-label]",
            "a[href*='/maps/place/']",
        ]
        best = None
        best_score = 0.0

        for selector in selectors:
            try:
                items = page.locator(selector)
                for index in range(min(items.count(), 120)):
                    item = items.nth(index)
                    try:
                        aria = self._normalise_text(item.get_attribute("aria-label") or "")
                        title = self._normalise_text(item.get_attribute("title") or "")
                        text = self._normalise_text(item.inner_text(timeout=250) or "")
                        href = item.get_attribute("href") or ""
                    except Exception:
                        continue
                    label = max((aria, title, text), key=len, default="")
                    if not label:
                        label = self._place_name_from_url(href)
                    score = self._name_match_score(requested_name, label)
                    if score > best_score:
                        best_score = score
                        best = (item, href)
            except Exception:
                continue

        if best is None or best_score < 0.80:
            return False

        item, href = best
        try:
            if href:
                page.goto(urljoin(str(page.url or ""), href), wait_until="domcontentloaded", timeout=self.timeout_ms)
            else:
                item.click(timeout=5_000)
            page.wait_for_timeout(2_000)
            self._dismiss_consent(page)
            page.wait_for_timeout(500)
            return self._place_pane_matches(page, requested_name)
        except Exception:
            return False

    def _assert_no_google_challenge(self, page) -> None:
        title = self._normalise_text(page.title())
        body = ""
        try:
            body = self._normalise_text(page.locator("body").inner_text(timeout=2_000))
        except Exception:
            pass
        challenge = f"{title} {body[:2500]}".lower()
        if "unusual traffic" in challenge or "not a robot" in challenge:
            raise RuntimeError(
                "Google displayed a verification/CAPTCHA page. Try again later or "
                "complete the browser check manually."
            )

    def _navigate_to_place(self, page, place_url: str) -> None:
        """Open the exact place URL first; use search only as a recovery path.

        The previous version deliberately opened a generated ``/maps/search/`` URL
        before the user's exact ``/maps/place/`` URL. For hotels in dense areas this
        produced a sponsored/result list, which was incorrectly accepted because it
        contained generic tab elements. Reviews could then never be opened. The exact
        URL is now authoritative and a result list is never treated as a place pane.
        """
        requested_name = self._place_name_from_url(place_url)
        search_url = self._search_navigation_url(place_url)
        last_error: Optional[Exception] = None

        # 1) The exact URL supplied by the user is the source of truth.
        try:
            page.goto(place_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            page.wait_for_timeout(2_200)
            self._dismiss_consent(page)
            page.wait_for_timeout(700)
            self._assert_no_google_challenge(page)

            if self._place_pane_matches(page, requested_name) or self._reviews_visible(page):
                return

            # Occasionally Google rewrites even an exact URL into a search list.
            if self._open_matching_search_result(page, requested_name):
                return
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc

        # 2) Search is only a fallback. It must explicitly select the matching result.
        if search_url:
            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                page.wait_for_timeout(2_500)
                self._dismiss_consent(page)
                page.wait_for_timeout(700)
                self._assert_no_google_challenge(page)

                if self._place_pane_matches(page, requested_name):
                    return
                if self._open_matching_search_result(page, requested_name):
                    return
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = exc

        raise ConnectionError(
            "Google Maps opened a search-results page but could not select the exact "
            f"place '{requested_name or 'requested location'}'. Use the full Google Maps "
            "place URL and try again."
        ) from last_error

    def _cdp_url(self) -> str:
        """Return an optional Chrome DevTools endpoint for an already-open browser.

        When set, the scraper attaches to a normal Chrome window instead of launching
        and closing its own browser. This lets the user sign in manually without ever
        storing an email or password in the project.
        """
        return os.environ.get("GOOGLE_MAPS_CDP_URL", "").strip()

    def _connect_cdp_context(self, playwright):
        cdp_url = self._cdp_url()
        if not cdp_url:
            return None, None

        try:
            browser = playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to the existing Chrome session at "
                f"{cdp_url}. Start scripts/start_google_maps_chrome.py first. "
                f"Original error: {exc}"
            ) from exc

        contexts = browser.contexts
        context = contexts[0] if contexts else None
        if context is None:
            raise RuntimeError(
                "Chrome CDP connection succeeded, but no browser context was available."
            )
        return browser, context

    def _profile_dir(self) -> Path:
        """Return a dedicated persistent browser profile for Google Maps.

        This intentionally does not use the user's normal Chrome profile. The dedicated
        directory stores cookies/local storage so a manual Google sign-in can be reused
        across scraper runs. Keep this directory private and out of source control.
        """
        configured = os.environ.get("GOOGLE_MAPS_PROFILE_DIR", "").strip()
        if configured:
            profile_dir = Path(configured).expanduser()
        else:
            profile_dir = Path.cwd() / ".browser_profiles" / "google_maps"
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir.resolve()

    def _persistent_profile_enabled(self) -> bool:
        raw = os.environ.get("GOOGLE_MAPS_PERSISTENT_PROFILE", "true").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _launch_persistent_context(self, playwright):
        """Launch Chrome/Edge with a dedicated profile that preserves login cookies."""
        user_data_dir = self._profile_dir()
        common = {
            "user_data_dir": str(user_data_dir),
            "headless": self.headless,
            "slow_mo": self.slow_mo_ms,
            "locale": self.locale,
            "viewport": {"width": 1440, "height": 1000},
        }

        # Prefer locally installed branded browsers, because they work even when the
        # Playwright Chromium download is unavailable. Bundled Chromium is the fallback.
        launch_attempts = [
            {**common, "channel": "chrome"},
            {**common, "channel": "msedge"},
        ]

        explicit_path = os.environ.get("GOOGLE_CHROME_PATH", "").strip()
        candidate_paths = [
            explicit_path,
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            str(Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]
        seen = set()
        for executable in candidate_paths:
            if not executable or executable in seen:
                continue
            seen.add(executable)
            if Path(executable).exists():
                launch_attempts.append({**common, "executable_path": executable})

        launch_attempts.append(common)

        errors = []
        for kwargs in launch_attempts:
            try:
                return playwright.chromium.launch_persistent_context(**kwargs)
            except Exception as exc:
                errors.append(str(exc))

        detail = errors[-1] if errors else "No persistent Chrome launch method succeeded."
        raise RuntimeError(
            "Could not launch the persistent Google Maps browser profile. Close any other "
            "scraper Chrome window using the same profile, or set GOOGLE_CHROME_PATH. "
            "Last error: " + detail
        )

    def _launch_browser(self, playwright):
        """Launch bundled Chromium, branded Chrome/Edge, or a known local executable."""
        launch_attempts = [
            {"headless": self.headless, "slow_mo": self.slow_mo_ms, "args": ["--disable-dev-shm-usage"]},
            {"channel": "chrome", "headless": self.headless, "slow_mo": self.slow_mo_ms},
            {"channel": "msedge", "headless": self.headless, "slow_mo": self.slow_mo_ms},
        ]

        explicit_path = os.environ.get("GOOGLE_CHROME_PATH", "").strip()
        candidate_paths = [
            explicit_path,
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
        for executable in candidate_paths:
            if executable and Path(executable).exists():
                launch_attempts.append(
                    {
                        "executable_path": executable,
                        "headless": self.headless,
                        "slow_mo": self.slow_mo_ms,
                        "args": ["--disable-dev-shm-usage"],
                    }
                )

        errors = []
        for kwargs in launch_attempts:
            try:
                return playwright.chromium.launch(**kwargs)
            except Exception as exc:
                errors.append(str(exc))

        detail = errors[-1] if errors else "No Chromium launch method succeeded."
        raise RuntimeError(
            "Could not launch Chromium/Chrome. Run: python -m playwright install chromium, "
            "or set GOOGLE_CHROME_PATH to your Chrome executable. Last error: " + detail
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape_reviews(
        self,
        place_url: str,
        max_reviews: int = 100,
        sort_order: str = "most_relevant",
        domain: str = "auto",
    ) -> Dict[str, Any]:
        url = self.validate_place_url(place_url)
        max_reviews = max(1, min(int(max_reviews or 100), 1_000))

        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise ImportError(
                "Playwright is not installed. Run: pip install playwright. The scraper can use installed Chrome/Edge."
            ) from exc

        collected: Dict[str, Dict[str, Any]] = {}
        browser = None
        context = None
        page = None
        external_cdp = False

        with sync_playwright() as playwright:
            if self._cdp_url():
                browser, context = self._connect_cdp_context(playwright)
                external_cdp = True
                # Always use a separate tab so the user's existing Maps tab is untouched.
                page = context.new_page()
            elif self._persistent_profile_enabled():
                context = self._launch_persistent_context(playwright)
                page = context.pages[0] if context.pages else context.new_page()
            else:
                browser = self._launch_browser(playwright)
                context = browser.new_context(
                    locale=self.locale,
                    viewport={"width": 1440, "height": 1000},
                )
                page = context.new_page()

            page.set_default_timeout(self.timeout_ms)

            try:
                self._navigate_to_place(page, url)

                entity = self._extract_place_metadata(page, domain)
                try:
                    self._open_reviews_panel(page)
                except Exception:
                    self._save_debug_artifacts(page, reason="reviews_panel")
                    raise
                selected_sort = self._apply_sort(page, sort_order)

                # The panel can report itself as open before review cards finish rendering.
                # Give the current layout a short window to expose data-review-id/text nodes.
                wait_until = time.time() + 10
                while time.time() < wait_until:
                    try:
                        if page.locator(
                            "div[data-review-id], span[jsname='bN97Pc'], "
                            "span[jsname='fbQN7e'], div.MyEned span.wiI7pd"
                        ).count() > 0:
                            break
                    except Exception:
                        pass
                    page.wait_for_timeout(300)

                feed = self._find_feed(page)
                root = feed if feed is not None else page

                no_growth_rounds = 0
                previous_card_count = 0
                max_rounds = min(500, max(25, (max_reviews // 4) + 30))

                for _ in range(max_rounds):
                    if not self._is_google_maps_page(page):
                        raise RuntimeError(
                            "The scraper left Google Maps while collecting reviews. "
                            "No external page will be scraped."
                        )

                    cards = self._review_cards(page, feed)
                    card_count = cards.count()

                    for index in range(card_count):
                        if len(collected) >= max_reviews:
                            break
                        try:
                            row = self._extract_card(cards.nth(index), entity, entity["resolved_url"])
                            if row:
                                collected[row["review_id"]] = row
                        except Exception:
                            continue

                    if len(collected) >= max_reviews:
                        break

                    if card_count <= previous_card_count:
                        no_growth_rounds += 1
                    else:
                        no_growth_rounds = 0
                    previous_card_count = max(previous_card_count, card_count)

                    if no_growth_rounds >= 6:
                        break

                    try:
                        if feed is not None:
                            feed.evaluate("el => { el.scrollTop = el.scrollHeight; }")
                        else:
                            page.mouse.wheel(0, 7_000)
                    except Exception:
                        try:
                            page.keyboard.press("End")
                        except Exception:
                            pass

                    page.wait_for_timeout(900)

                if not collected:
                    debug_paths = self._save_debug_artifacts(page, reason="no_review_text")
                    debug_hint = debug_paths.get("html") or debug_paths.get("screenshot") or ""
                    raise ValueError(
                        "The Reviews panel opened, but no written review text could be parsed. "
                        "This is a DOM-selector issue, not a login/credential issue. "
                        + (f"Debug file: {debug_hint}" if debug_hint else "")
                    )

                rows = list(collected.values())[:max_reviews]
                raw_df = pd.DataFrame(rows)
                common_df = pd.DataFrame(
                    [
                        {
                            "review_id": row["review_id"],
                            "domain": row["domain"],
                            "entity_id": row["entity_id"],
                            "entity_name": row["entity_name"],
                            "review_text": row["review_text"],
                            "rating": row["rating"] if row["rating"] is not None else 3.0,
                            "rating_original": row["rating"] if row["rating"] is not None else "",
                            "review_date": row["review_date"],
                            "source": "google_maps_playwright",
                            "raw_source_path": "",
                        }
                        for row in rows
                    ],
                    columns=self.COMMON_COLUMNS,
                )

                metadata = {
                    **entity,
                    "requested_reviews": max_reviews,
                    "actual_scraped_reviews": len(common_df),
                    "sort_order_requested": sort_order,
                    "sort_order_applied": selected_sort,
                    "headless": self.headless,
                }
                self.last_metadata = metadata
                return {"raw_df": raw_df, "common_df": common_df, "metadata": metadata}

            except PlaywrightTimeoutError as exc:
                raise TimeoutError(
                    "Google Maps did not finish loading before the scraper timeout. "
                    "Check the URL/internet connection and try again."
                ) from exc
            finally:
                if external_cdp:
                    # The Chrome window is owned by the user. Close only the scraper tab
                    # and disconnect Playwright; never close the signed-in browser/profile.
                    try:
                        if page is not None:
                            page.close()
                    except Exception:
                        pass
                    # Do not call browser.close() here: this Chrome session was
                    # started by the user and must remain open after Playwright detaches.
                else:
                    try:
                        if context is not None:
                            context.close()
                    except Exception:
                        pass
                    try:
                        if browser is not None:
                            browser.close()
                    except Exception:
                        pass

    def scrape_to_files(
        self,
        place_url: str,
        output_dir: str | Path,
        max_reviews: int = 100,
        sort_order: str = "most_relevant",
        domain: str = "auto",
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = self.scrape_reviews(
            place_url=place_url,
            max_reviews=max_reviews,
            sort_order=sort_order,
            domain=domain,
        )

        metadata = result["metadata"]
        safe_name = self._safe_filename(metadata.get("entity_name", "google_maps_place"))
        raw_path = output_path / f"raw_google_maps_{safe_name}.csv"
        prepared_path = output_path / f"prepared_google_maps_{safe_name}.csv"
        metadata_path = output_path / f"google_maps_{safe_name}_metadata.json"

        result["raw_df"].to_csv(raw_path, index=False, encoding="utf-8-sig")

        common_df = result["common_df"].copy()
        common_df["raw_source_path"] = str(raw_path)
        common_df.to_csv(prepared_path, index=False, encoding="utf-8-sig")

        import json

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "prepared_csv": str(prepared_path),
            "raw_csv": str(raw_path),
            "metadata_json": str(metadata_path),
            "domain": metadata["domain"],
            "entity_id": metadata["entity_id"],
            "entity_name": metadata["entity_name"],
            "category": metadata.get("category", ""),
            "overall_rating": metadata.get("overall_rating"),
            "displayed_review_count": metadata.get("displayed_review_count"),
            "scraped_count": int(metadata["actual_scraped_reviews"]),
            "resolved_url": metadata["resolved_url"],
            "sort_order_applied": metadata["sort_order_applied"],
        }

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.last_metadata)
