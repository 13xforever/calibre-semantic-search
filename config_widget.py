'''Settings dialog: embeddings, indexing, attribute schema.'''

from __future__ import annotations

from qt.core import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
)

from calibre.utils.localization import _

from .utils import AttrField, Settings


class SettingsWidget(QDialog):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.setWindowTitle(_('Semantic search settings'))
        self.resize(760, 520)
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

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._collect_and_accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)

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
