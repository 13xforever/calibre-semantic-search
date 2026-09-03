'''Semantic Search plugin for calibre.

Adds meaning-based search across your library: books are chunked into
paragraph groups with chapter context, embedded via an OpenAI-compatible
embedding server, and searched by vector similarity. LLM attribute extraction
(e.g. protagonist gender, romance tropes) lands in calibre custom columns so
you can combine semantic results with normal metadata searches.
'''

from calibre.customize import InterfaceActionBase


class SemanticSearch(InterfaceActionBase):
    name = 'Semantic Search'
    description = 'Semantic (meaning-based) search across your library, with LLM-extracted book attributes in custom columns'
    version = (1, 0, 0)
    author = 'CalibreSemanticIndex'
    minimum_calibre_version = (8, 0, 0)

    actual_plugin = 'calibre_plugins.semantic_search.gui:SemanticSearchAction'
