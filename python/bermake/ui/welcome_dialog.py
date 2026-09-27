"""The welcome dialog (M7.7): pick a template and units, or open a file.

The first QDialog in the application, so it sets the convention for the ones
after it. The rule that matters for the test suite: exec() is never called from
application logic. This class exposes its selection as properties, MainWindow is
the only caller that execs it, and tests construct it, drive the widgets and read
the result.

Everything worth testing is in apply_template, which is a module-level function
over DocumentSettings and needs no dialog at all.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from bermake.document import DocumentSettings
from bermake.templates import TEMPLATES, Template, template_for_key
from bermake.ui import preferences
from bermake.units import UnitSystem

# Label shown, unit passed to DocumentSettings.set_metric. "in" routes to
# set_imperial instead, which takes no unit string.
_UNIT_CHOICES: tuple[tuple[str, str], ...] = (
    ("Millimetres", "mm"),
    ("Centimetres", "cm"),
    ("Metres", "m"),
    ("Inches", "in"),
)

_IMPERIAL_UNIT = "in"

# The radio-button column never needs more than a short template name, so it
# gets a fixed budget and the description column takes the rest (Task 6c
# Finding 1's fix: a QGridLayout with no stretch set sizes both columns to
# their content, which clips the longer descriptions instead of wrapping
# them).
_RADIO_COLUMN_MIN_WIDTH = 160
_DESCRIPTION_COLUMN_PADDING = 48


def apply_template(doc: DocumentSettings, template: Template, unit: str) -> None:
    """Apply `template` to `doc`, with `unit` overriding the template's own.

    Order matters and is fixed: the template's full Units first, then the
    override. set_metric and set_imperial both preserve the fields they do not
    name, so overriding the unit keeps the template's metric_precision and
    imperial_denominator. Rebuilding Units from scratch here would silently
    reset precision to the dataclass default.
    """
    doc.set_units(template.units)
    if unit == _IMPERIAL_UNIT:
        doc.set_imperial(template.units.imperial_denominator)
    else:
        doc.set_metric(unit)
    doc.set_environment(template.environment)


class WelcomeDialog(QDialog):
    """Template grid, units selector, Open, and a do-not-show checkbox."""

    def __init__(self, settings: QSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to Bermake")
        self._settings = settings
        self._wants_open = False

        stored = preferences.read_default_template(settings)
        initial = template_for_key(stored)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Start a new model"))

        grid = QGridLayout()
        # The description column takes the slack; the radio-button column
        # stays at its content width (Task 6c Finding 1).
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        self._template_buttons: dict[str, QRadioButton] = {}
        self._description_labels: dict[str, QLabel] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for row, template in enumerate(TEMPLATES):
            button = QRadioButton(template.name)
            button.setToolTip(template.description)
            button.setChecked(template.key == initial.key)
            self._group.addButton(button)
            self._template_buttons[template.key] = button
            grid.addWidget(button, row, 0)
            description = QLabel(template.description)
            description.setTextFormat(Qt.TextFormat.PlainText)
            description.setWordWrap(True)
            grid.addWidget(description, row, 1)
            self._description_labels[template.key] = description
        layout.addLayout(grid)
        self._template_grid = grid
        self.setMinimumWidth(self._minimum_width_for_descriptions())

        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Units"))
        self._units_combo = QComboBox()
        for label, unit in _UNIT_CHOICES:
            self._units_combo.addItem(label, unit)
        self._units_combo.setCurrentIndex(self._index_for_template(initial))
        units_row.addWidget(self._units_combo)
        units_row.addStretch(1)
        layout.addLayout(units_row)

        self._show_checkbox = QCheckBox("Show this window on startup")
        self._show_checkbox.setChecked(preferences.read_show_welcome(settings))
        layout.addWidget(self._show_checkbox)

        buttons = QHBoxLayout()
        open_button = QPushButton("Open...")
        open_button.clicked.connect(self._on_open_clicked)
        buttons.addWidget(open_button)
        buttons.addStretch(1)
        start_button = QPushButton("Start modelling")
        start_button.setDefault(True)
        start_button.clicked.connect(self.accept)
        buttons.addWidget(start_button)
        layout.addLayout(buttons)

    def _minimum_width_for_descriptions(self) -> int:
        """Wide enough that the longest description wraps to at most two lines.

        Half its single-line width is a safe two-line budget -- word wrap
        reflows well before a column gets exactly that narrow, so this leaves
        slack rather than landing right on the wrap boundary. Padding covers
        layout spacing and the label's own margins; the radio-button column
        gets a fixed budget of its own since template names are short and
        fixed, unlike the descriptions this fix is about.
        """
        metrics = self.fontMetrics()
        longest = max((template.description for template in TEMPLATES), key=len)
        single_line_width = metrics.horizontalAdvance(longest)
        description_width = single_line_width // 2 + _DESCRIPTION_COLUMN_PADDING
        return description_width + _RADIO_COLUMN_MIN_WIDTH

    @staticmethod
    def _index_for_template(template: Template) -> int:
        """The units-combo row matching a template's own unit."""
        wanted = (
            _IMPERIAL_UNIT
            if template.units.system is UnitSystem.IMPERIAL
            else template.units.metric_unit
        )
        for index, (_label, unit) in enumerate(_UNIT_CHOICES):
            if unit == wanted:
                return index
        return 0

    @property
    def selected_template(self) -> Template:
        for key, button in self._template_buttons.items():
            if button.isChecked():
                return template_for_key(key)
        return template_for_key(None)

    @property
    def selected_unit(self) -> str:
        return str(self._units_combo.currentData())

    @property
    def show_on_startup(self) -> bool:
        return bool(self._show_checkbox.isChecked())

    @property
    def wants_open(self) -> bool:
        """True when the user asked to open a file instead of starting new.

        Reported rather than acted on: MainWindow owns the open-file flow, and a
        second copy of it here would drift from the real one.
        """
        return self._wants_open

    def _on_open_clicked(self) -> None:
        self._wants_open = True
        self.accept()
