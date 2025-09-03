# Migration from PySimpleGUI to PySide6

## Overview

This document tracks the migration of the SEET project's GUI components from PySimpleGUI to PySide6. The migration affects only the sensitivity analysis application located in `seet/sensitivity_analysis/sensitivity_analysis_app/sensitivity_analysis_app.py`.

## PySimpleGUI Usage Analysis

### Confirmation of Scope
✅ **Confirmed**: PySimpleGUI usage is exclusive to `seet/sensitivity_analysis/sensitivity_analysis_app/sensitivity_analysis_app.py`

No other Python files in the project import or use PySimpleGUI.

### PySimpleGUI Widgets and Functions Used

#### Core Widgets
- `sg.Window` - Main application window
- `sg.Frame` - Grouping container with border and title
- `sg.Tab` - Tab widget for organizing content
- `sg.TabGroup` - Container for multiple tabs

#### Input Widgets
- `sg.InputText` - Single-line text input
- `sg.Input` - Generic input field
- `sg.Slider` - Slider control for numeric values
- `sg.Check` - Checkbox widget

#### Display Widgets
- `sg.Text` - Static text label
- `sg.Button` - Clickable button

#### Layout and Spacing
- `sg.Push` - Pushes elements to one side (similar to stretch)
- `sg.HorizontalSeparator` - Horizontal line separator

#### File/Folder Selection
- `sg.FileBrowse` - File browser button
- `sg.FolderBrowse` - Folder browser button

#### Dialog Functions
- `sg.popup_get_folder()` - Folder selection dialog
- `sg.Popup()` - Simple message popup
- `sg.one_line_progress_meter()` - Progress bar dialog

#### Settings and Persistence
- `sg.user_settings_get_entry()` - Retrieve user settings
- `sg.user_settings_set_entry()` - Store user settings

#### Constants and Events
- `sg.WIN_CLOSED` - Window close event constant

#### Low-level Access
- `sg.tk` - Direct access to underlying Tkinter objects (used for DPI scaling fix)

### Widget Usage Patterns

#### Slider Creation Pattern
```python
sg.Slider(
    range=(min, max),
    default_value=max/2,
    resolution=(max - min) / 20,
    orientation="vertical",
    size=(10, 10),
    key=key
)
```

#### Frame Layout Pattern
```python
sg.Frame(title, layout, border_width=0)
```

#### Tab Creation Pattern
```python
sg.Tab(title, [[content_layout]])
```

#### Input with Browse Pattern
```python
sg.Input(default_value, size=(36, 1), key=key),
sg.FileBrowse(target=key)
```

## Current Migration Progress

### ✅ Completed Steps:
1. **Application Structure Migration**:
   - Replaced PySimpleGUI imports with PySide6 imports
   - Changed class inheritance: `SensitivityAnalysisAPP()` → `SensitivityAnalysisAPP(QMainWindow)`
   - Implemented Qt application lifecycle with proper `__init__()` and `closeEvent()`
   - Added QSettings for user preferences persistence

2. **Helper Methods Migration**:
   - `_labeled_slider()`: `sg.Slider` → `QSlider` with `QGroupBox` container
   - `_multi_slider_widget()`: `sg.Frame` → `QGroupBox` with `QHBoxLayout`
   - `create_tab_widget()`: Basic structure adapted for Qt widgets

### 🔄 Next Steps:
1. Migrate the `create_window()` method - the main GUI construction
2. Replace all remaining `sg.*` calls with PySide6 equivalents
3. Implement file browse functionality with `QFileDialog`
4. Handle event processing and user settings

## Migration Tasks

### Phase 1: Setup and Planning
- [x] Install PySide6 dependencies
- [x] Create basic PySide6 application structure
- [ ] Set up Qt Designer files (optional)

### Phase 2: Core Application Structure
- [x] Migrate main window (`sg.Window` → `QMainWindow`)
- [x] Migrate event loop (preparation: removed PySimpleGUI, added Qt structure)
- [x] Migrate window close handling (`sg.WIN_CLOSED` → `closeEvent`)

### Phase 3: Widget Migration - IN PROGRESS
- [x] Helper method: `_labeled_slider` (`sg.Slider` → `QSlider` in `QGroupBox`)
- [x] Helper method: `_multi_slider_widget` (`sg.Frame` → `QGroupBox` with `QHBoxLayout`)
- [x] Helper method: `create_tab_widget` (basic structure)
- [ ] Migrate frames (`sg.Frame` → `QGroupBox`)
- [ ] Migrate tabs (`sg.Tab`/`sg.TabGroup` → `QTabWidget`)
- [ ] Migrate text labels (`sg.Text` → `QLabel`)
- [ ] Migrate input fields (`sg.Input`/`sg.InputText` → `QLineEdit`)
- [ ] Migrate buttons (`sg.Button` → `QPushButton`)
- [ ] Migrate checkboxes (`sg.Check` → `QCheckBox`)

### Phase 4: Layout Management
- [ ] Migrate layout system (lists → `QVBoxLayout`/`QHBoxLayout`)
- [ ] Migrate push/stretch (`sg.Push` → `QSpacerItem`)
- [ ] Migrate separators (`sg.HorizontalSeparator` → `QFrame`)

### Phase 5: File/Folder Dialogs
- [ ] Migrate file browser (`sg.FileBrowse` → `QFileDialog`)
- [ ] Migrate folder browser (`sg.FolderBrowse` → `QFileDialog`)
- [ ] Migrate popup folder selector (`sg.popup_get_folder` → `QFileDialog`)

### Phase 6: Dialogs and Feedback
- [ ] Migrate message popups (`sg.Popup` → `QMessageBox`)
- [ ] Migrate progress meter (`sg.one_line_progress_meter` → `QProgressDialog`)

### Phase 7: Settings and Persistence
- [ ] Migrate user settings (`sg.user_settings_*` → `QSettings`)

### Phase 8: Integration and Testing
- [ ] Remove PySimpleGUI imports
- [ ] Update requirements.txt
- [ ] Test all functionality
- [ ] Fix DPI scaling issues (remove Tkinter workaround)

## Notes

### Dependencies
- Current: `PySimpleGUI` (Tkinter-based)
- Target: `PySide6` (Qt6-based)

### Key Challenges
1. **Event System**: PySimpleGUI's blocking `read()` vs Qt's event-driven model
2. **Layout System**: PySimpleGUI's list-based layouts vs Qt's layout managers
3. **Settings Persistence**: PySimpleGUI's built-in settings vs QSettings
4. **DPI Scaling**: Current Tkinter/PyPlot scaling fix needs Qt equivalent

## PySimpleGUI to PySide6 Mapping Table

| PySimpleGUI | PySide6 |
|-------------|---------|
| `sg.Window` | `QMainWindow` or `QWidget` |
| `sg.Frame` | `QGroupBox` |
| `sg.Tab` | `QWidget` (content) |
| `sg.TabGroup` | `QTabWidget` |
| `sg.InputText` | `QLineEdit` |
| `sg.Input` | `QLineEdit` |
| `sg.Slider` | `QSlider` |
| `sg.Check` | `QCheckBox` |
| `sg.Text` | `QLabel` |
| `sg.Button` | `QPushButton` |
| `sg.Push` | `QSpacerItem` with `QSizePolicy.Expanding` |
| `sg.HorizontalSeparator` | `QFrame` with `QFrame.HLine` |
| `sg.FileBrowse` | `QPushButton` + `QFileDialog.getOpenFileName()` |
| `sg.FolderBrowse` | `QPushButton` + `QFileDialog.getExistingDirectory()` |
| `sg.popup_get_folder()` | `QFileDialog.getExistingDirectory()` |
| `sg.Popup()` | `QMessageBox.information()` |
| `sg.one_line_progress_meter()` | `QProgressDialog` |
| `sg.user_settings_get_entry()` | `QSettings.value()` |
| `sg.user_settings_set_entry()` | `QSettings.setValue()` |
| `sg.WIN_CLOSED` | `QCloseEvent` in `closeEvent()` method |
| `sg.tk` (Tkinter access) | Direct Qt widget access (no wrapper needed) |

### Widget Property Mappings

| PySimpleGUI Property | PySide6 Equivalent |
|---------------------|-------------------|
| `size=(width, height)` | `setFixedSize(width, height)` or `setMinimumSize()` |
| `key="identifier"` | `setObjectName("identifier")` |
| `enable_events=True` | Connect to appropriate signal (e.g., `clicked`, `textChanged`) |
| `default_value` | `setText()`, `setValue()`, or `setChecked()` |
| `range=(min, max)` | `setRange(min, max)` |
| `orientation="vertical"` | `Qt.Vertical` |
| `border_width=0` | `setStyleSheet("border: 0px;")` |
| `target=key` | Manual connection between widgets (see explanation below) |

### Layout System Mapping

| PySimpleGUI Layout | PySide6 Layout |
|-------------------|----------------|
| `[[widget1, widget2]]` | `QHBoxLayout` with `addWidget()` |
| `[[widget1], [widget2]]` | `QVBoxLayout` with `addWidget()` |
| Nested lists | Nested layout managers |
| `sg.Push()` | `addStretch()` or `QSpacerItem` |

### Target Property Explanation

The `target=key` property in PySimpleGUI is used by browse buttons (`sg.FileBrowse`, `sg.FolderBrowse`) to automatically update another widget when a file/folder is selected.

**PySimpleGUI Example:**
```python
sg.Input(key="-FILENAME-", size=(36, 1)),
sg.FileBrowse(target="-FILENAME-")
```
When the user selects a file via FileBrowse, PySimpleGUI automatically puts the filename into the Input widget with key `"-FILENAME-"`.

**PySide6 Equivalent:**
In PySide6, this automatic connection doesn't exist. You must manually connect the file dialog result to the target widget:

```python
# Create widgets
filename_input = QLineEdit()
browse_button = QPushButton("Browse")

# Manual connection via signal/slot
def on_browse_clicked():
    filename, _ = QFileDialog.getOpenFileName(self, "Select File")
    if filename:
        filename_input.setText(filename)

browse_button.clicked.connect(on_browse_clicked)
```

This is what "manual connection between widgets" means - you explicitly handle the relationship between widgets through Qt's signal/slot mechanism rather than relying on an automatic `target` property.

### Architecture Impact
The migration will maintain the same functional interface but completely change the underlying GUI framework, requiring updates to:
- Widget creation and configuration
- Event handling mechanisms
- Layout management
- File/folder dialog interactions
