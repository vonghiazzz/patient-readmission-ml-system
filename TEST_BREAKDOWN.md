# Test structure and breakdown

## Current verified totals

| Measurement | Count |
| --- | ---: |
| Python test function definitions | 34 |
| Test items collected and executed by pytest | 38 |
| Additional items created by parametrization | 4 |
| Test modules containing executable tests | 9 |

`pytest --collect-only` is the source of truth for executable test items. The reporting utility
uses Python's AST for function definitions, so strings and documentation containing test-like text
are not counted as functions.

## Breakdown by file

| Test file | Function definitions | Pytest items |
| --- | ---: | ---: |
| `tests/data/test_preprocessing.py` | 3 | 3 |
| `tests/data/test_splitting.py` | 2 | 2 |
| `tests/data/test_validation.py` | 7 | 7 |
| `tests/integration/test_api.py` | 7 | 7 |
| `tests/integration/test_failure_startup.py` | 1 | 1 |
| `tests/production/test_production_contract.py` | 5 | 5 |
| `tests/unit/test_evaluation.py` | 3 | 3 |
| `tests/unit/test_operations_config.py` | 3 | 3 |
| `tests/unit/test_schemas.py` | 3 | 7 |
| **Total** | **34** | **38** |

`tests/test_breakdown.py` is a reporting utility and contains no executable test function, so it is
not included in the nine test modules above.

## Why 34 functions become 38 items

The function
`test_notebook_excluded_or_out_of_contract_values_are_rejected` in
`tests/unit/test_schemas.py` uses five parameter sets:

```text
1 function definition -> 5 pytest items
```

Compared with a normal single execution, this adds four items:

```text
34 definitions - 1 parametrized definition + 5 parametrized items = 38 items
```

## Commands

Show every definition and its exact collected item count:

```bash
python tests/test_breakdown.py
```

Run the complete project verification:

```bash
python tests/verify_project.py
```

Ask pytest directly:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

Expected summary for the current suite:

```text
34 test function definitions
38 test items collected
38 passed
```
