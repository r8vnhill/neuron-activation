# Chat rules

## Rule 1: Code documentation

Whenever you're asked for documentation you should provide the functions/classes with their respective docstrings
following doxygen syntax. If you're asked for a specific function/class you should provide the docstring for that
specific function/class. If you're asked for a specific module you should provide the docstring for that specific
module.

Remember that in python, class and function docstring is positioned right after the declaration, for example:

```python
def validate_alignment(alignment: str) -> bool:
    """
    Checks if an alignment string is valid.

    This function verifies if all characters in a given alignment string are
    either 'l', 'c', or 'r'. Spaces and '|' characters are ignored during the check.

    :param alignment: Alignment string to be validated.
    :return: Boolean indicating whether the alignment string is valid.
    """
    ...


@dataclass
class Alignment:
    """
    Class for LaTeX table column alignment specification.

    This class is used to encapsulate and validate LaTeX-compatible column
    alignment strings for table creation.
    """

    alignment: str

    def __init__(self, alignment: Union[str, 'Alignment']):
        """
        Initializes an Alignment instance.

        :param alignment: Column alignment string.
        :raises ValueError: If alignment string is invalid.
        """
        ...
```

# Rule 2: