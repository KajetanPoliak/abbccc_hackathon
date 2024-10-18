# ABB CCC Hackathon 2024
Repository for the ABB CCC Hackathon 2024

## Pre-Commits

For development, make sure that the [pre-commit](https://pypi.org/project/pre-commit/)
library is installed.
Then, run

```sh
# For each time one wants to run the pre-commit hooks
#  without committing
> ./pre_commit.sh
# Or run each command from the script separately
```

The **pre-commit** routine will be performed automatically whenever one
attempts to commit anything. The commit will fail if one of the hooks fails.
Therefore, manual runs of the *sh* script are not necessary.

To tweak the hooks of **pre-commit**, edit
[.pre-commit-config.yaml](.pre-commit-config.yaml).
The configuration of the [mypy](https://mypy.readthedocs.io/en/stable/)
static type checker hook can be found in [mypy.ini](mypy.ini).