"""Attach the user's clarified post-competition score semantics to metadata."""
from pathlib import Path
import json


def annotate_evaluation(metadata):
    context = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
    metadata['evaluation_context'] = context
    metadata['public_subset_only'] = context['public_subset_only']
    if metadata.get('exploratory_after_public_feedback'):
        metadata['exploratory_after_public_feedback'] = False
        metadata['exploratory_after_contest_feedback'] = True
    reference = metadata.get('current_public_reference')
    if reference is not None and 'public_score' in reference:
        reference['contest_score'] = reference['public_score']
        reference['evaluation_set'] = context['score_scope']
    for row in metadata.get('submissions', []):
        if 'public_score' in row:
            row['contest_score'] = row['public_score']
    return metadata
