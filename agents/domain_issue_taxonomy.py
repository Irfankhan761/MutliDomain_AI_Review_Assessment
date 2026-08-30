"""
Phase 8: Domain-aware issue taxonomy.

This file defines the issues that the Semantic Issue Mining Agent can detect
for each review domain. Each issue has:
- a human-readable label
- semantic descriptions used for embedding matching
- keywords used for evidence phrase extraction and confidence boosting
- severity score used later by the Risk Scoring Agent
"""

DOMAIN_ISSUE_TAXONOMY = {
    "mobile_app": {
        "crash": {
            "label": "Crash / App Failure",
            "severity_score": 25,
            "descriptions": [
                "app crashes, app freezes, app force closes, application stops working",
                "the app does not open, keeps closing, crashes after update",
                "software failure, unstable app, unusable mobile application"
            ],
            "keywords": [
                "crash", "crashes", "crashed", "freezes", "frozen", "force close",
                "stops working", "not open", "won't open", "doesn't open", "unusable"
            ],
        },
        "login": {
            "label": "Login / Account Access",
            "severity_score": 20,
            "descriptions": [
                "login failure, cannot sign in, password issue, account access problem",
                "OTP not received, verification issue, authentication failed",
                "unable to access account or user profile"
            ],
            "keywords": [
                "login", "log in", "sign in", "password", "otp", "verification",
                "account", "authentication", "cannot access", "can't access"
            ],
        },
        "payment": {
            "label": "Payment / Billing",
            "severity_score": 25,
            "descriptions": [
                "payment failed, money deducted, billing problem, transaction issue",
                "charged incorrectly, purchase failed, subscription payment problem",
                "refund not received after payment or transaction failure"
            ],
            "keywords": [
                "payment", "paid", "charged", "billing", "transaction", "money deducted",
                "purchase", "refund", "card", "bank"
            ],
        },
        "privacy": {
            "label": "Privacy / Data Concern",
            "severity_score": 25,
            "descriptions": [
                "privacy concern, data tracking, permissions, personal data risk",
                "app collects location, contacts, private information or user data",
                "security and privacy issue in mobile application"
            ],
            "keywords": [
                "privacy", "permission", "permissions", "tracking", "data", "location",
                "contacts", "personal information", "security"
            ],
        },
        "subscription": {
            "label": "Subscription / Cancellation",
            "severity_score": 18,
            "descriptions": [
                "subscription complaint, cancellation problem, auto renewal issue",
                "premium plan cannot be cancelled, recurring payment concern",
                "charged for subscription after cancellation"
            ],
            "keywords": [
                "subscription", "subscribe", "premium", "renewal", "cancel", "cancelled",
                "auto renewal", "trial"
            ],
        },
        "advertising": {
            "label": "Advertisements / Popups",
            "severity_score": 12,
            "descriptions": [
                "too many ads, pop up advertisements, intrusive advertising",
                "annoying ads interrupting app use",
                "advertisement overload affecting user experience"
            ],
            "keywords": [
                "ad", "ads", "advertisement", "advertisements", "popup", "pop up",
                "pop-up", "commercial"
            ],
        },
        "inappropriate_content": {
            "label": "Inappropriate / Explicit Content",
            "severity_score": 25,
            "descriptions": [
                "unwanted sexual, explicit or adult content shown in a mobile application",
                "complaint about sexual, obscene, nude or inappropriate pictures or videos",
                "user wants to avoid shameful, sexual or inappropriate visual content",
                "mobile application exposes users to unwanted explicit or adult material"
            ],
            "keywords": [
                "sex", "sex content", "sexy", "sexy pictures", "sexual", "sexual content",
                "explicit", "explicit content", "adult", "adult content", "adult material",
                "inappropriate", "inappropriate content", "inappropriate pictures",
                "shameful", "shamefull", "shameful pictures", "shamefull pictures",
                "obscene", "porn", "pornographic", "nude", "nudity", "naked", "naked pictures"
            ],
        },
        "performance": {
            "label": "Performance / Speed",
            "severity_score": 15,
            "descriptions": [
                "app is slow, lagging, loading problem, poor performance",
                "application response is delayed, slow loading, performance issue",
                "mobile app runs slowly or becomes unresponsive"
            ],
            "keywords": [
                "slow", "lag", "laggy", "loading", "performance", "delay", "delayed",
                "unresponsive", "hang"
            ],
        },
    },

    "hotel": {
        "cleanliness": {
            "label": "Cleanliness",
            "severity_score": 25,
            "descriptions": [
                "dirty room, unclean bathroom, bad smell, dust, mold or poor hygiene",
                "hotel cleanliness problem, room was not cleaned properly",
                "unclean bed, toilet, bathroom, carpet or public area"
            ],
            "keywords": [
                "dirty", "unclean", "smell", "smelly", "bathroom", "toilet", "dust",
                "mold", "mould", "stain", "stains", "hygiene", "cleanliness"
            ],
        },
        "room_quality": {
            "label": "Room Quality",
            "severity_score": 18,
            "descriptions": [
                "poor room quality, small room, broken furniture, uncomfortable bed",
                "air conditioning problem, room facilities not working",
                "hotel room was old, damaged, hot, cold or poorly maintained"
            ],
            "keywords": [
                "room", "bed", "air condition", "air conditioning", "ac", "broken",
                "small", "hot", "cold", "old", "maintenance", "shower"
            ],
        },
        "staff_service": {
            "label": "Staff / Service",
            "severity_score": 18,
            "descriptions": [
                "rude staff, poor service, unhelpful reception, bad customer service",
                "hotel staff were not helpful, reception problem, manager complaint",
                "service quality problem in hotel stay"
            ],
            "keywords": [
                "staff", "service", "rude", "unhelpful", "reception", "manager",
                "customer service", "front desk", "attitude"
            ],
        },
        "noise": {
            "label": "Noise",
            "severity_score": 12,
            "descriptions": [
                "noisy room, loud music, traffic noise, poor soundproofing",
                "sleep disturbed by noise, loud neighbours or street noise",
                "hotel stay affected by noise disturbance"
            ],
            "keywords": [
                "noise", "noisy", "loud", "sound", "traffic", "music", "disturb",
                "disturbed", "soundproof"
            ],
        },
        "booking": {
            "label": "Booking / Check-in",
            "severity_score": 15,
            "descriptions": [
                "booking problem, reservation issue, check-in delay, cancellation issue",
                "hotel reservation not found, wrong booking, check out problem",
                "booking process or check-in experience was poor"
            ],
            "keywords": [
                "booking", "reservation", "check in", "check-in", "checkout", "check out",
                "cancel", "cancelled", "room not ready"
            ],
        },
        "wifi": {
            "label": "Wi-Fi / Internet",
            "severity_score": 10,
            "descriptions": [
                "wifi problem, internet not working, poor connection",
                "slow internet, no Wi-Fi, unstable network in hotel",
                "hotel internet connectivity issue"
            ],
            "keywords": [
                "wifi", "wi-fi", "internet", "connection", "network", "signal"
            ],
        },
    },

    "ecommerce": {
        "delivery": {
            "label": "Delivery / Shipping",
            "severity_score": 15,
            "descriptions": [
                "late delivery, shipping problem, item did not arrive, delivery delay",
                "product arrived late, shipment issue, courier problem",
                "delivery experience was poor or delayed"
            ],
            "keywords": [
                "delivery", "delivered", "shipping", "shipment", "late", "arrived",
                "courier", "tracking", "not received"
            ],
        },
        "refund": {
            "label": "Refund / Return",
            "severity_score": 22,
            "descriptions": [
                "refund problem, return refused, money not returned",
                "customer could not return item, refund delayed, return policy issue",
                "refund and return complaint after online purchase"
            ],
            "keywords": [
                "refund", "return", "returned", "money back", "replacement", "exchange",
                "can't return", "cannot return"
            ],
        },
        "fake_product": {
            "label": "Fake / Counterfeit Product",
            "severity_score": 25,
            "descriptions": [
                "fake product, counterfeit item, not original, duplicate product",
                "product is not genuine, misleading listing, fake brand item",
                "customer received a fake or counterfeit product"
            ],
            "keywords": [
                "fake", "counterfeit", "not original", "duplicate", "not genuine",
                "knockoff", "copy"
            ],
        },
        "damaged_item": {
            "label": "Damaged / Broken Item",
            "severity_score": 20,
            "descriptions": [
                "damaged item, broken product, cracked, leaked or defective",
                "product arrived damaged, bottle leaked, item was broken",
                "customer received a defective or damaged item"
            ],
            "keywords": [
                "damaged", "broken", "cracked", "leaked", "leaking", "defective",
                "faulty", "spill", "spilled"
            ],
        },
        "packaging": {
            "label": "Packaging",
            "severity_score": 10,
            "descriptions": [
                "poor packaging, package damaged, box opened, bad packing",
                "item packaging was weak or product was not packed properly",
                "packaging quality issue in online order"
            ],
            "keywords": [
                "package", "packaging", "box", "packed", "packing", "sealed", "seal"
            ],
        },
        "product_quality": {
            "label": "Product Quality",
            "severity_score": 18,
            "descriptions": [
                "poor product quality, cheap material, product does not work",
                "bad quality item, not as described, product performance issue",
                "customer is dissatisfied with product quality"
            ],
            "keywords": [
                "quality", "poor", "cheap", "bad product", "not as described",
                "doesn't work", "does not work", "useless"
            ],
        },
    },

    "restaurant": {
        "food_quality": {
            "label": "Food Quality",
            "severity_score": 18,
            "descriptions": [
                "bad food quality, poor taste, cold food, stale food",
                "meal was not fresh, food tasted bad, dish quality problem",
                "customer complaint about food taste or freshness"
            ],
            "keywords": [
                "food", "taste", "tasted", "cold", "stale", "fresh", "meal",
                "dish", "flavour", "flavor", "overcooked", "undercooked"
            ],
        },
        "hygiene": {
            "label": "Hygiene / Cleanliness",
            "severity_score": 25,
            "descriptions": [
                "restaurant hygiene problem, dirty table, unclean place, hair in food",
                "poor cleanliness, dirty restaurant, food safety concern",
                "hygiene or cleanliness complaint in restaurant"
            ],
            "keywords": [
                "dirty", "hygiene", "clean", "unclean", "hair", "table", "toilet",
                "bathroom", "smell"
            ],
        },
        "staff_service": {
            "label": "Staff / Service",
            "severity_score": 18,
            "descriptions": [
                "rude staff, poor restaurant service, bad waiter or server",
                "slow service, unhelpful staff, poor customer service",
                "restaurant service quality complaint"
            ],
            "keywords": [
                "staff", "service", "rude", "waiter", "server", "manager",
                "customer service", "attitude"
            ],
        },
        "wait_time": {
            "label": "Waiting Time",
            "severity_score": 12,
            "descriptions": [
                "long wait time, slow service, order delayed, waiting too long",
                "restaurant delay, food took too long to arrive",
                "customer waited too long for order or table"
            ],
            "keywords": [
                "wait", "waiting", "slow", "delay", "delayed", "took long",
                "long time", "queue"
            ],
        },
        "price_value": {
            "label": "Price / Value",
            "severity_score": 10,
            "descriptions": [
                "expensive restaurant, overpriced food, poor value for money",
                "price is too high, not worth the money, bad value",
                "customer complaint about price or value"
            ],
            "keywords": [
                "price", "expensive", "overpriced", "value", "money", "worth",
                "cost", "costly"
            ],
        },
        "ambience": {
            "label": "Ambience / Atmosphere",
            "severity_score": 8,
            "descriptions": [
                "bad ambience, poor atmosphere, uncomfortable seating or music",
                "restaurant place was not comfortable, bad environment",
                "ambience or atmosphere complaint"
            ],
            "keywords": [
                "ambience", "ambiance", "atmosphere", "music", "place",
                "seating", "environment", "decor"
            ],
        },
    },
}


GLOBAL_PROBLEM_CUES = [
    "bad", "poor", "terrible", "awful", "worst", "horrible", "issue", "problem",
    "problems", "complaint", "complaints", "failed", "failure", "not working",
    "doesn't work", "does not work", "dirty", "broken", "damaged", "refund",
    "return", "crash", "crashes", "rude", "slow", "delay", "delayed", "fake",
    "charged", "payment", "privacy", "unable", "cannot", "can't", "won't",
    "disappointed", "waste", "useless", "leaked", "leaking", "stale", "cold",
    "overpriced", "unhelpful", "missing", "cancel", "cancelled",
    "sex content", "sexy", "sexual", "sexual content", "explicit", "explicit content",
    "adult content", "adult material", "inappropriate", "inappropriate content",
    "shameful", "shamefull", "obscene", "porn", "pornographic",
    "nude", "nudity", "naked", "naked pictures"
]


def get_domain_taxonomy(domain: str) -> dict:
    """Returns taxonomy for selected domain. Defaults to mobile_app if unknown."""
    if domain in DOMAIN_ISSUE_TAXONOMY:
        return DOMAIN_ISSUE_TAXONOMY[domain]
    return DOMAIN_ISSUE_TAXONOMY["mobile_app"]


def get_issue_severity_score(domain: str, issue: str) -> int:
    taxonomy = get_domain_taxonomy(domain)
    if issue in taxonomy:
        return int(taxonomy[issue].get("severity_score", 0))
    return 0


def severity_level_from_score(score: int) -> str:
    if score >= 25:
        return "high"
    if score >= 15:
        return "medium"
    if score > 0:
        return "low"
    return "none"
