"""Site customizations for the local Streamlit app.

This patches Streamlit's SessionClient Protocol __init__ to be a harmless no-op.
It fixes a Tornado/Streamlit MRO issue where RequestHandler.__init__ calls
super().__init__(), which hits Protocol's _no_init_or_replace_init and raises
TypeError during websocket setup.
"""

try:
    import streamlit.runtime.session_manager as _sm

    def _session_client_init(self, *args, **kwargs):
        # Protocol classes are not meant to be instantiated. When used as a mixin
        # in Streamlit's BrowserWebSocketHandler, Tornado's super().__init__ chain
        # hits this method. A no-op keeps the chain safe.
        return None

    _sm.SessionClient.__init__ = _session_client_init  # type: ignore[assignment]
except Exception:
    # If Streamlit isn't available, do nothing.
    pass
