"""Keep user-facing selection deliverables in the project's documentation tree."""
from pathlib import Path


def review_directory(directory: Path) -> Path:
    directory = Path(directory).absolute()
    for root in directory.parents:
        if (root / 'pyproject.toml').is_file():
            analysis = root / 'workspace' / 'analysis-runs'
            if directory.is_relative_to(analysis):
                relative = directory.relative_to(analysis)
                target = root / 'docs' / '高质量Trace筛选_20260915'
                if relative.parts[0] != 'selection-summaries':
                    target /= 'batch-reports'
                target /= relative
                target.mkdir(parents=True, exist_ok=True)
                return target
            break
    return directory
