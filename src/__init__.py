'''Semantic Search plugin for calibre.

Adds meaning-based search across your library: books are chunked into
paragraph groups with chapter context, embedded via an OpenAI-compatible
embedding server, and searched by vector similarity. LLM attribute extraction
(e.g. protagonist gender, romance tropes) lands in calibre custom columns so
you can combine semantic results with normal metadata searches.
'''

from calibre.customize import InterfaceActionBase

from . import utils as _utils


def _bootstrap_external_deps():
    # Import the module (not the name) and guard with getattr: calibre's in-place
    # plugin reload re-runs this file but does NOT re-import submodules, so a
    # stale/partial utils must degrade to "no external deps", never crash the load.
    cleanup = getattr(_utils, 'process_pending_removals', None)
    if callable(cleanup):
        try:
            cleanup()  # delete files a prior in-session uninstall deferred (nothing imported yet)
        except Exception as e:
            print(f'semantic search: pending dependency cleanup failed: {e!r}')
    fn = getattr(_utils, 'bootstrap_external_deps', None)
    if callable(fn):
        try:
            fn()  # make the external folder importable before store/gui load (no-op if absent)
        except Exception as e:
            print(f'semantic search: external dependency bootstrap failed: {e!r}')


_bootstrap_external_deps()


class SemanticSearch(InterfaceActionBase):
    name = 'Semantic Search'
    description = 'Semantic (meaning-based) search across your library, with LLM-extracted book attributes in custom columns'
    version = (1, 0, 0)
    author = 'CalibreSemanticIndex'
    minimum_calibre_version = (8, 0, 0)

    actual_plugin = 'calibre_plugins.semantic_search.gui:SemanticSearchAction'
