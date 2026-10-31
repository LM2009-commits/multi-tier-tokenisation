"""Tokenisers: WLT (word level), CLT (character level) and the combined MTT.

Note: this sub-package is called ``tokenizers`` like the HuggingFace library. Inside
``wlt.py`` the line ``from tokenizers import ...`` is an absolute import and resolves to
the HuggingFace library, as long as the ``mtt/`` directory itself is never put on
``sys.path`` (always run things from the project root as ``python -m mtt.xxx``).
"""
