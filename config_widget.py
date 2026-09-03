'''Settings dialog: embeddings, indexing, attribute schema.'''

from __future__ import annotations

from qt.core import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QEvent,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QObject,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    Qt,
)

from calibre.utils.localization import _

from .utils import AttrField, Settings


class _HelpFilter(QObject):
    """Updates the help box when the mouse enters a bound widget or its label."""

    def __init__(self, box: QTextEdit):
        super().__init__()
        self.box = box

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Enter:
            text = getattr(obj, '_help_text', None)
            if text:
                self.box.setPlainText(text)
        return False


HELP_DEFAULT = _('Hover over an option to see what it does.')

HELP_SERVER_URL = _(
    'Base URL of an OpenAI-compatible embedding server (the plugin calls POST /v1/embeddings).\n'
    'Ollama: http://localhost:11434 (default)\n'
    'LM Studio: http://localhost:1234/v1\n'
    'vLLM / Unsloth: the root URL of your server.'
)

HELP_MODEL = _(
    'Name/ID of the embedding model served by that server (e.g. an Ollama tag).\n\n'
    'Recommended models:\n'
    '- nomic-embed-text — good default, fast even on CPU\n'
    '- Qwen3-Embedding-4B or bge-m3 — best quality; want a GPU with ~8GB+ VRAM\n'
    '- bge-small-en-v1.5 / mxbai-embed-large — lighter alternatives\n\n'
    'Note: changing the model changes vector dimensions and scores. After switching models, '
    "run 'Re-index all books' from the Semantic search menu so every book shares one model."
)

HELP_API_KEY = _(
    "Optional. Leave empty for local servers such as Ollama. Set it if your server requires an "
    "'Authorization: Bearer <key>' header (some vLLM / LM Studio setups)."
)

HELP_BATCH = _(
    'How many text chunks to send per embeddings request. Larger values are faster but use more RAM '
    'on the server. 64 works well; lower it if the server runs out of memory.'
)

HELP_CONCURRENCY = _(
    'Number of embedding requests to run in parallel. Higher values speed up initial indexing on '
    'multi-core servers or GPUs; keep at 1-2 for small models running on CPU.'
)

HELP_TIMEOUT = _('Per-request timeout in seconds. Raise it if indexing stalls on a slow CPU server.')

HELP_BACKEND = _(
    "Where chunk vectors are stored.\n"
    "- auto (default): LanceDB if the 'lancedb' Python package is installed, otherwise SQLite\n"
    '- sqlite: always available; vectors live in a file next to metadata.db\n'
    "- lancedb: requires 'pip install lancedb'\n\n"
    'Changing the backend or the embedding model requires re-indexing for consistent results.'
)

HELP_FORMATS = _(
    'Which file format to extract text from, one per line, top to bottom. The first format a book has '
    'is used. EPUB/AZW3 first gives the best structure; keep PDF last because its text extraction is '
    'lower quality.'
)

HELP_TARGET = _(
    'Approximate size of each search chunk in characters (default 1000). Larger chunks give more '
    'context but coarser matches; smaller chunks are the reverse. Changing this requires re-indexing.'
)

HELP_OVERLAP = _(
    'Number of characters repeated between adjacent chunks, so sentences that fall on a chunk '
    'boundary can still be matched.'
)

HELP_MAX_CHUNKS = _(
    'Safety cap on the number of chunks per book (0 = unlimited). Useful to keep indexing time and '
    'disk usage down for very long books.'
)

HELP_ATTR_MODE = _(
    'sampled: one LLM call over an evenly spaced sample of the book (~12k chars) — fast and cheap, '
    'good for most books.\n'
    'fulltext: map-reduce over the entire text — slower and uses more tokens, but catches details '
    'that only appear deep in long books.'
)

HELP_ATTR_ENABLED = _(
    'Include this field in attribute extraction. Its calibre custom column is created/updated automatically.'
)
HELP_ATTR_NAME = _(
    "Internal name of the field (lowercase letters, digits, underscores). Determines the calibre "
    "custom column (label 'ss_...') where values are stored."
)
HELP_ATTR_TYPE = _(
    "text = single value (e.g. 'first person'). tags = multi-value list, stored like #tags so you can "
    'filter on individual values.'
)
HELP_ATTR_DESC = _(
    'Shown to the LLM as guidance for what to extract. Be specific about the expected format and give examples.'
)
HELP_ATTR_TABLE = _(
    "Attribute fields extracted by the LLM into calibre custom columns. Toggle 'Enabled', edit names, "
    "types and descriptions, then run 'Extract attributes...' from the Semantic search menu. Changing "
    'the schema marks affected books for re-extraction.'
)


class SettingsWidget(QDialog):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.setWindowTitle(_('Semantic search settings'))
        self.resize(760, 610)
        v = QVBoxLayout(self)
        tabs = QTabWidget()
        v.addWidget(tabs)

        # -- Embeddings tab ----------------------------------------------------
        emb_tab = QWidget()
        f = QFormLayout(emb_tab)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.e_base_url = QLineEdit(self.s.embed.base_url)
        self.e_model = QLineEdit(self.s.embed.model)
        self.e_api_key = QLineEdit(self.s.embed.api_key)
        self.e_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.e_batch = QSpinBox()
        self.e_batch.setRange(1, 512)
        self.e_batch.setValue(self.s.embed.batch_size)
        self.e_conc = QSpinBox()
        self.e_conc.setRange(1, 32)
        self.e_conc.setValue(self.s.embed.concurrency)
        self.e_timeout = QSpinBox()
        self.e_timeout.setRange(5, 3600)
        self.e_timeout.setValue(self.s.embed.timeout)
        f.addRow(_('Server URL (OpenAI-compatible):'), self.e_base_url)
        f.addRow(_('Model name:'), self.e_model)
        f.addRow(_('API key (optional):'), self.e_api_key)
        f.addRow(_('Batch size:'), self.e_batch)
        f.addRow(_('Concurrency:'), self.e_conc)
        f.addRow(_('Timeout (s):'), self.e_timeout)
        tabs.addTab(emb_tab, _('Embeddings'))

        # -- Indexing tab --------------------------------------------------------
        idx_tab = QWidget()
        f2 = QFormLayout(idx_tab)
        f2.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.i_backend = QComboBox()
        self.i_backend.addItems(['auto', 'sqlite', 'lancedb'])
        self.i_backend.setCurrentText(self.s.vector_backend)
        self.i_formats = QPlainTextEdit('\n'.join(self.s.format_priority))
        self.i_formats.setPlaceholderText(_('One format per line, in priority order'))
        self.i_target = QSpinBox()
        self.i_target.setRange(200, 10000)
        self.i_target.setValue(self.s.target_chars)
        self.i_overlap = QSpinBox()
        self.i_overlap.setRange(0, 2000)
        self.i_overlap.setValue(self.s.overlap_chars)
        self.i_maxchunks = QSpinBox()
        self.i_maxchunks.setRange(0, 100000)
        self.i_maxchunks.setValue(self.s.max_chunks_per_book)
        self.i_maxchunks.setSpecialValueText(_('Unlimited'))
        self.i_attrmode = QComboBox()
        self.i_attrmode.addItems(['sampled', 'fulltext'])
        self.i_attrmode.setCurrentText(self.s.attr_mode)
        f2.addRow(_('Vector backend:'), self.i_backend)
        f2.addRow(_('Format priority:'), self.i_formats)
        f2.addRow(_('Target chunk size (chars):'), self.i_target)
        f2.addRow(_('Overlap (chars):'), self.i_overlap)
        f2.addRow(_('Max chunks per book:'), self.i_maxchunks)
        f2.addRow(_('Attribute extraction mode:'), self.i_attrmode)
        tabs.addTab(idx_tab, _('Indexing'))

        # -- Attributes tab ------------------------------------------------------
        att_tab = QWidget()
        av = QVBoxLayout(att_tab)
        self.attr_table = QTableWidget(len(self.s.attributes), 4)
        self.attr_table.setHorizontalHeaderLabels([_('Enabled'), _('Name'), _('Type'), _('Description')])
        self.attr_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for r, a in enumerate(self.s.attributes):
            cb = QTableWidgetItem()
            cb.setCheckState(Qt.CheckState.Unchecked if not a.enabled else Qt.CheckState.Checked)
            self.attr_table.setItem(r, 0, cb)
            self.attr_table.setItem(r, 1, QTableWidgetItem(a.name))
            self.attr_table.setItem(r, 2, QTableWidgetItem(a.type))
            self.attr_table.setItem(r, 3, QTableWidgetItem(a.description))
        av.addWidget(self.attr_table, 1)
        btns = QHBoxLayout()
        b_add = QPushButton(_('Add field'))
        b_add.clicked.connect(lambda: self._add_attr())
        b_del = QPushButton(_('Remove selected'))
        b_del.clicked.connect(self._del_attr)
        btns.addStretch(1)
        btns.addWidget(b_del)
        btns.addWidget(b_add)
        av.addLayout(btns)
        tabs.addTab(att_tab, _('Attributes'))

        # -- help box + per-option hints -----------------------------------------
        self.help_box = QTextEdit()
        self.help_box.setReadOnly(True)
        self.help_box.setMaximumHeight(80)
        self.help_box.setPlainText(HELP_DEFAULT)
        v.addWidget(self.help_box)

        self._help_filter = _HelpFilter(self.help_box)
        B = self._bind_help
        B(HELP_SERVER_URL, self.e_base_url, f.labelForField(self.e_base_url))
        B(HELP_MODEL, self.e_model, f.labelForField(self.e_model))
        B(HELP_API_KEY, self.e_api_key, f.labelForField(self.e_api_key))
        B(HELP_BATCH, self.e_batch, f.labelForField(self.e_batch))
        B(HELP_CONCURRENCY, self.e_conc, f.labelForField(self.e_conc))
        B(HELP_TIMEOUT, self.e_timeout, f.labelForField(self.e_timeout))
        B(HELP_BACKEND, self.i_backend, f2.labelForField(self.i_backend))
        B(HELP_FORMATS, self.i_formats, f2.labelForField(self.i_formats))
        B(HELP_TARGET, self.i_target, f2.labelForField(self.i_target))
        B(HELP_OVERLAP, self.i_overlap, f2.labelForField(self.i_overlap))
        B(HELP_MAX_CHUNKS, self.i_maxchunks, f2.labelForField(self.i_maxchunks))
        B(HELP_ATTR_MODE, self.i_attrmode, f2.labelForField(self.i_attrmode))
        for col, key in enumerate((HELP_ATTR_ENABLED, HELP_ATTR_NAME, HELP_ATTR_TYPE, HELP_ATTR_DESC)):
            item = self.attr_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(key)
        B(HELP_ATTR_TABLE, self.attr_table)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._collect_and_accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)

    def _bind_help(self, text: str, *widgets):
        """Give widgets a tooltip and make hovering them update the help box."""
        for w in widgets:
            if w is None:
                continue
            w._help_text = text
            w.setToolTip(text)
            w.setWhatsThis(text)
            w.installEventFilter(self._help_filter)

    # -- attribute table ---------------------------------------------------------

    def _add_attr(self):
        r = self.attr_table.rowCount()
        self.attr_table.insertRow(r)
        cb = QTableWidgetItem()
        cb.setCheckState(Qt.CheckState.Checked)
        self.attr_table.setItem(r, 0, cb)
        self.attr_table.setItem(r, 1, QTableWidgetItem('new_field'))
        self.attr_table.setItem(r, 2, QTableWidgetItem('text'))
        self.attr_table.setItem(r, 3, QTableWidgetItem(''))

    def _del_attr(self):
        r = self.attr_table.currentRow()
        if r >= 0:
            self.attr_table.removeRow(r)

    # -- collect -----------------------------------------------------------------

    def _collect_and_accept(self):
        s = Settings()
        s.embed.base_url = self.e_base_url.text().strip() or 'http://localhost:11434'
        s.embed.model = self.e_model.text().strip() or 'nomic-embed-text'
        s.embed.api_key = self.e_api_key.text().strip()
        s.embed.batch_size = self.e_batch.value()
        s.embed.concurrency = self.e_conc.value()
        s.embed.timeout = self.e_timeout.value()
        s.vector_backend = self.i_backend.currentText()
        fmts = [x.strip().upper() for x in self.i_formats.toPlainText().replace(',', '\n').splitlines() if x.strip()]
        s.format_priority = fmts or list(s.format_priority)
        s.target_chars = self.i_target.value()
        s.overlap_chars = self.i_overlap.value()
        s.max_chunks_per_book = self.i_maxchunks.value()
        s.attr_mode = self.i_attrmode.currentText()
        attrs = []
        for r in range(self.attr_table.rowCount()):
            name = (self.attr_table.item(r, 1).text() if self.attr_table.item(r, 1) else '').strip().lower()
            name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
            if not name:
                continue
            typ = (self.attr_table.item(r, 2).text() if self.attr_table.item(r, 2) else 'text').strip().lower()
            if typ not in ('text', 'tags'):
                typ = 'text'
            desc = (self.attr_table.item(r, 3).text() if self.attr_table.item(r, 3) else '').strip()
            enabled = bool(self.attr_table.item(r, 0) and self.attr_table.item(r, 0).checkState() == Qt.CheckState.Checked)
            # preserve label from existing schema when possible
            old = next((a for a in self.s.attributes if a.name == name), None)
            label = old.label if old else ('ss_' + name[:30])
            attrs.append(AttrField(name=name, label=label, type=typ, description=desc, enabled=enabled))
        s.attributes = attrs or [a.clone() for a in self.s.attributes]
        self._result = s
        self.accept()

    def settings(self) -> Settings:
        return getattr(self, '_result', self.s)
