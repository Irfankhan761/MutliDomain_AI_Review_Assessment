from pathlib import Path
import argparse
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from agents.groq_final_summary_agent import GroqFinalSummaryAgent


def main():
    parser = argparse.ArgumentParser(description='Phase 12: Groq final summary test.')
    parser.add_argument('--review-results', default='outputs/phase11_explainability_summary_pipeline/multidomain_review_level_results.csv')
    parser.add_argument('--entity-summary', default='outputs/phase11_explainability_summary_pipeline/multidomain_entity_level_summary.csv')
    parser.add_argument('--entity-id', default=None)
    parser.add_argument('--entity-name', default=None)
    parser.add_argument('--domain', default=None)
    parser.add_argument('--output', default='outputs/phase12_groq_final_summary/groq_final_report.txt')
    args = parser.parse_args()

    review_path = PROJECT_ROOT / args.review_results
    entity_path = PROJECT_ROOT / args.entity_summary
    output_path = PROJECT_ROOT / args.output

    if not review_path.exists():
        raise FileNotFoundError(f'Review-level results not found: {review_path}')
    if not entity_path.exists():
        raise FileNotFoundError(f'Entity summary not found: {entity_path}')

    review_df = pd.read_csv(review_path, low_memory=False)
    entity_df = pd.read_csv(entity_path, low_memory=False)

    print('\nPHASE 12B: GROQ FINAL SUMMARY AGENT')
    print('=' * 80)
    print('Review results:', review_path)
    print('Entity summary:', entity_path)
    print('Review rows:', len(review_df))
    print('Entities:', len(entity_df))

    agent = GroqFinalSummaryAgent(env_path=PROJECT_ROOT / '.env')
    result = agent.generate_report(review_df, entity_df, args.entity_id, args.entity_name, args.domain)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result['final_report'], encoding='utf-8')
    context_path = output_path.with_name('groq_context_payload.json')
    context_path.write_text(result['context_payload'], encoding='utf-8')

    print('\nSelected entity:')
    selected = result['selected_entity']
    for key in ['domain', 'entity_type', 'entity_id', 'entity_name', 'average_trust_score', 'overall_risk_level', 'top_issues']:
        print(f'{key}: {selected.get(key)}')

    print('\nFINAL GROQ REPORT')
    print('=' * 80)
    print(result['final_report'])
    print('\nSaved:')
    print(output_path)
    print(context_path)


if __name__ == '__main__':
    main()
