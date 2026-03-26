"use strict";

const el = {
  datasetDirInput: document.getElementById("datasetDirInput"),
  modelPathInput: document.getElementById("modelPathInput"),
  labelsDirInput: document.getElementById("labelsDirInput"),
  classesFileInput: document.getElementById("classesFileInput"),
  classesInput: document.getElementById("classesInput"),
  sam2ConfigInput: document.getElementById("sam2ConfigInput"),
  videoPathInput: document.getElementById("videoPathInput"),
  videoStrideInput: document.getElementById("videoStrideInput"),
  videoMaxFramesInput: document.getElementById("videoMaxFramesInput"),
  carryPrevMaskInput: document.getElementById("carryPrevMaskInput"),
  browseVideoBtn: document.getElementById("browseVideoBtn"),
  videoInfo: document.getElementById("videoInfo"),
  languageSelect: document.getElementById("languageSelect"),
  engineSelect: document.getElementById("engineSelect"),
  samModelTypeInput: document.getElementById("samModelTypeInput"),
  loadSessionBtn: document.getElementById("loadSessionBtn"),
  browseDatasetBtn: document.getElementById("browseDatasetBtn"),
  browseModelBtn: document.getElementById("browseModelBtn"),
  browseSam2ConfigBtn: document.getElementById("browseSam2ConfigBtn"),
  browseLabelsBtn: document.getElementById("browseLabelsBtn"),
  browseClassesBtn: document.getElementById("browseClassesBtn"),
  confInput: document.getElementById("confInput"),
  minAreaInput: document.getElementById("minAreaInput"),
  pointsPerSideInput: document.getElementById("pointsPerSideInput"),
  pointsPerBatchInput: document.getElementById("pointsPerBatchInput"),
  predIouThreshInput: document.getElementById("predIouThreshInput"),
  stabilityScoreThreshInput: document.getElementById("stabilityScoreThreshInput"),
  stabilityScoreOffsetInput: document.getElementById("stabilityScoreOffsetInput"),
  boxNmsThreshInput: document.getElementById("boxNmsThreshInput"),
  cropNLayersInput: document.getElementById("cropNLayersInput"),
  cropNmsThreshInput: document.getElementById("cropNmsThreshInput"),
  cropOverlapRatioInput: document.getElementById("cropOverlapRatioInput"),
  cropNPointsDownscaleInput: document.getElementById("cropNPointsDownscaleInput"),
  mergeSameClassInput: document.getElementById("mergeSameClassInput"),
  nextUnlabeledBtn: document.getElementById("nextUnlabeledBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  imageSelect: document.getElementById("imageSelect"),
  imageCounter: document.getElementById("imageCounter"),
  predictBtn: document.getElementById("predictBtn"),
  saveBtn: document.getElementById("saveBtn"),
  saveAllBtn: document.getElementById("saveAllBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  zoomFitBtn: document.getElementById("zoomFitBtn"),
  zoomLabel: document.getElementById("zoomLabel"),
  canvasWrap: document.getElementById("canvasWrap"),
  canvas: document.getElementById("annotCanvas"),
  statusBar: document.getElementById("statusBar"),
  classSelect: document.getElementById("classSelect"),
  deleteBtn: document.getElementById("deleteBtn"),
  clearBtn: document.getElementById("clearBtn"),
  pointModeBtn: document.getElementById("pointModeBtn"),
  clearPointsBtn: document.getElementById("clearPointsBtn"),
  segmentPointsBtn: document.getElementById("segmentPointsBtn"),
  propagateVideoBtn: document.getElementById("propagateVideoBtn"),
  refinePointsInput: document.getElementById("refinePointsInput"),
  mergeContoursInput: document.getElementById("mergeContoursInput"),
  seedMaskForPropagateInput: document.getElementById("seedMaskForPropagateInput"),
  boxList: document.getElementById("boxList"),
  removeUncheckedBtn: document.getElementById("removeUncheckedBtn"),
  checkAllBtn: document.getElementById("checkAllBtn"),
  pickerModal: document.getElementById("pickerModal"),
  pickerTitle: document.getElementById("pickerTitle"),
  pickerCloseBtn: document.getElementById("pickerCloseBtn"),
  pickerRootsBtn: document.getElementById("pickerRootsBtn"),
  pickerUpBtn: document.getElementById("pickerUpBtn"),
  pickerSelectCurrentBtn: document.getElementById("pickerSelectCurrentBtn"),
  pickerPath: document.getElementById("pickerPath"),
  pickerList: document.getElementById("pickerList"),
};

const ctx = el.canvas.getContext("2d");

const PICKER_CONFIG = {
  dataset: { titleKey: "picker.select_dataset", mode: "dir", input: el.datasetDirInput, allowFolder: true },
  model: { titleKey: "picker.select_model", mode: "model", input: el.modelPathInput, allowFile: true },
  sam2Config: { titleKey: "picker.select_sam2_yaml", mode: "yaml", input: el.sam2ConfigInput, allowFile: true },
  labels: { titleKey: "picker.select_labels", mode: "dir", input: el.labelsDirInput, allowFolder: true },
  classes: { titleKey: "picker.select_classes", mode: "yaml", input: el.classesFileInput, allowFile: true },
  video: { titleKey: "picker.select_video", mode: "video", input: el.videoPathInput, allowFile: true },
};

const state = {
  sessionLoaded: false,
  lang: "en",
  images: [],
  imageIndex: -1,
  imagePath: "",
  image: null,
  classes: [],
  defaultClassId: 0,
  segments: [],
  annotationsByPath: {},
  annotationCountByPath: {},
  dirtyPaths: new Set(),
  selectedIndex: -1,
  interaction: null,
  dirty: false,
  historyByPath: {},
  loadToken: 0,
  mode: "image",
  video: { active: false, info: null },
  view: { width: 0, height: 0, fitScale: 1, scale: 1, offsetX: 0, offsetY: 0, userZoom: 1, lastPointer: null },
  picker: { target: null, currentPath: null, parentPath: null, lastData: null },
  draw: { active: false, points: [], preview: null },
  points: { active: false, items: [] },
};

const LANG_STORAGE_KEY = "segmentit_ui_lang";

const I18N = {
  en: {
    "app.title": "SegmentIt",
    "app.brand": "SegmentIt",
    "app.subtitle": "Interactive segmentation and fast mask labeling.",
    "lang.label": "Language",
    "lang.en": "English",
    "lang.ru": "Russian",
    "common.browse": "Browse",
    "left.section.data_source": "Data source",
    "left.label.dataset_dir": "Dataset directory",
    "left.label.video_path": "Video (MP4/AVI/MOV)",
    "left.label.frame_stride": "Frame stride",
    "left.label.max_frames": "Max frames",
    "left.label.model_path": "Segmentation model path (optional)",
    "left.label.sam2_config": "SAM 2 config YAML (optional)",
    "left.label.engine": "Segmentation engine",
    "left.label.sam_model_type": "SAM model type (vit_h / vit_l / vit_b)",
    "left.advanced.summary": "Auto segmentation (SAM settings)",
    "left.advanced.points_per_side": "Points per side",
    "left.advanced.points_per_batch": "Points per batch",
    "left.advanced.pred_iou_thresh": "Pred IoU thresh",
    "left.advanced.stability_thresh": "Stability thresh",
    "left.advanced.stability_offset": "Stability offset",
    "left.advanced.box_nms": "Box NMS",
    "left.advanced.crop_layers": "Crop layers",
    "left.advanced.crop_nms": "Crop NMS",
    "left.advanced.crop_overlap": "Crop overlap",
    "left.advanced.crop_points_scale": "Crop points scale",
    "left.advanced.merge_same_class": "Merge same class masks",
    "left.label.labels_dir": "Labels output directory (optional)",
    "left.label.classes_file": "Classes YAML/TXT file (optional)",
    "left.label.classes": "Classes (one per line)",
    "left.button.load": "Load",
    "left.video_info.not_loaded": "Video not loaded.",
    "left.video_info.details": "Frames: {frameCount} | FPS: {fps} | {width}x{height} | stride {stride}",
    "left.hint.tip":
      "Tip: set either a dataset folder or a video path, then click Load. For SAM2, a model path is required.",
    "left.controls.title": "Controls:",
    "left.controls.select": "Left click: select mask",
    "left.controls.negative_point": "Shift/Alt + left click: negative point",
    "left.controls.pan": "Middle click: pan",
    "left.controls.add_vertex": "Add vertex: <code>Ctrl</code> + click on edge",
    "left.controls.draw": "Draw: press <code>N</code>, add points, <code>Enter</code> to finish",
    "left.controls.delete": "Delete: <code>Del</code> or the Delete mask button",
    "main.nav.prev": "Prev",
    "main.nav.next": "Next",
    "main.nav.next_unlabeled": "Next unlabeled",
    "main.actions.auto_segment": "Auto segment",
    "main.actions.save": "Save",
    "main.actions.save_all": "Save all",
    "main.label.conf": "Conf (YOLO)",
    "main.label.min_area": "Min area",
    "main.zoom.fit": "Fit",
    "status.ready": "Ready.",
    "right.title": "Masks",
    "right.label.class": "Class for segmentation/drawing",
    "right.section.points": "Points (SAM2)",
    "right.button.point_mode": "Point mode",
    "right.button.clear_points": "Clear points",
    "right.button.segment_points": "Segment from points",
    "right.button.propagate_video": "Propagate video",
    "right.checkbox.refine": "Refine selected mask (keep points)",
    "right.checkbox.merge_contours": "Merge contours into single mask",
    "right.checkbox.seed_mask": "Use current mask for propagation",
    "right.checkbox.copy_prev": "Copy masks from previous frame",
    "right.hint.negative_point": "Shift/Alt: negative point",
    "right.hint.enter_backspace": "<code>Enter</code> &mdash; segment, <code>Backspace</code> &mdash; remove last point",
    "right.button.delete_mask": "Delete mask",
    "right.button.clear_masks": "Clear masks",
    "right.box_list_title": "Masks",
    "right.button.remove_unchecked": "Remove unchecked",
    "right.button.check_all": "Check all",
    "right.hotkeys.title": "Hotkeys",
    "right.hotkeys.save": "<code>Ctrl+S</code> &mdash; save",
    "right.hotkeys.save_all": "<code>Ctrl+Shift+S</code> &mdash; save all",
    "right.hotkeys.auto": "<code>P</code> &mdash; auto segment",
    "right.hotkeys.draw": "<code>N</code> &mdash; draw mask",
    "right.hotkeys.point_mode": "<code>T</code> &mdash; point mode",
    "right.hotkeys.insert_vertex": "<code>Ctrl</code> + click edge &mdash; insert vertex",
    "right.hotkeys.nav": "<code>Left/Right</code> &mdash; prev/next",
    "right.hotkeys.undo": "<code>Ctrl+Z</code> &mdash; undo",
    "right.hotkeys.zoom": "<code>+ / -</code> &mdash; zoom",
    "right.hotkeys.fit": "<code>0</code> &mdash; fit",
    "picker.title": "Select path",
    "picker.close": "Close",
    "picker.roots": "Roots",
    "picker.up": "Up",
    "picker.select_current": "Select current folder",
    "picker.select_dataset": "Select dataset folder",
    "picker.select_model": "Select model file",
    "picker.select_sam2_yaml": "Select SAM 2 YAML config",
    "picker.select_labels": "Select labels folder",
    "picker.select_classes": "Select classes YAML/TXT",
    "picker.select_video": "Select video file",
    "picker.tag.dir": "DIR",
    "picker.tag.file": "FILE",
    "picker.empty": "No files",
    "points.mode": "Point mode",
    "points.mode_on": "Point mode: ON",
    "segment.area": "Area {area} px",
    "status.initial": "Set a dataset folder or a video path, then click Load.",
    "status.current":
      "{path} | {index}/{total} | masks: {count} (kept {kept}){dirty} | unsaved: {unsaved}{tail}",
    "status.dirty_suffix": " (unsaved)",
    "status.extra.cached": "cached",
    "status.extra.copied_prev": "copied from previous frame",
    "status.extra.loaded": "loaded",
    "status.extra.saved_to": "Saved to {path}",
    "status.extra.saved_counts": "Saved: {saved}, errors {errors}",
    "status.point_mode_cleared": "Point mode: points cleared.",
    "status.point_mode_help": "Point mode: LMB adds, RMB removes.",
    "status.vertex_inserted": "Vertex inserted.",
    "status.draw_mode_help": "Draw mode: click to add points; double click or Enter to finish.",
    "status.undo_applied": "Undo applied.",
    "status.undo_none": "Nothing to undo.",
    "status.failed_load_roots": "Failed to load roots: {error}",
    "status.failed_list_path": "Failed to list path: {error}",
    "status.no_images_found": "No images found in the dataset.",
    "status.both_dataset_and_video": "Both dataset and video are set. Loading video.",
    "status.dataset_or_video_required": "Dataset folder or video path is required.",
    "status.dataset_required": "Dataset path is required.",
    "status.loading_dataset": "Loading dataset...",
    "status.failed_load_dataset": "Failed to load dataset: {error}",
    "status.video_path_required": "Video path is required.",
    "status.extracting_frames": "Extracting frames from video...",
    "status.video_loaded": "Video loaded. Use frames like images to annotate.",
    "status.failed_load_video": "Failed to load video: {error}",
    "status.loading_image": "Loading image...",
    "status.failed_load_image": "Failed to load image: {error}",
    "status.no_unlabeled_found": "No unlabeled images found.",
    "status.no_image_selected": "No image selected.",
    "status.running_auto_seg": "Running auto segmentation...",
    "status.segmentation_complete": "Segmentation complete: masks {count}. Review the list.",
    "status.segmentation_failed": "Segmentation failed: {error}",
    "status.add_points_first": "Add some points first.",
    "status.need_positive_point": "Add at least one positive point.",
    "status.sam2_required_select": "SAM 2 is required. Select it in the segmentation settings.",
    "status.sam2_yaml_required": "SAM 2 YAML config is required.",
    "status.sam2_model_required": "SAM 2 model path is required.",
    "status.segmenting_points": "Segmenting from points...",
    "status.no_masks_returned": "No masks returned.",
    "status.failed_choose_primary": "Failed to choose a primary mask to refine.",
    "status.mask_updated_points": "Mask updated from points. Add more points to refine.",
    "status.added_masks_points": "Added masks from points: {count}.",
    "status.point_seg_failed": "Point segmentation failed: {error}",
    "status.load_video_first": "Load a video first.",
    "status.sam2_required": "SAM 2 is required.",
    "status.sam2_yaml_required_prop": "SAM 2 YAML config is required for propagation.",
    "status.sam2_model_required_prop": "SAM 2 model path is required for propagation.",
    "status.propagating": "Propagating mask through video... this may take a while.",
    "status.propagation_complete": "Propagation complete: saved {saved}, errors {errors}.",
    "status.propagation_failed": "Propagation failed: {error}",
    "status.saving": "Saving...",
    "status.save_failed": "Save failed: {error}",
    "status.nothing_to_save": "Nothing to save.",
    "status.batch_saving": "Batch saving: {count}...",
    "status.batch_save_failed": "Batch save failed: {error}",
    "status.removed_unchecked_masks": "Removed unchecked masks: {count}.",
    "status.no_mask_selected_delete": "No mask selected to delete.",
    "error.invalid_response": "Invalid response: {text}",
    "error.request_failed": "Request failed ({status})",
    "error.failed_load_image": "Failed to load image.",
  },
  ru: {
    "app.title": "SegmentIt",
    "app.brand": "SegmentIt",
    "app.subtitle": "Интерактивная сегментация и быстрая разметка масок.",
    "lang.label": "Язык",
    "lang.en": "Английский",
    "lang.ru": "Русский",
    "common.browse": "Обзор",
    "left.section.data_source": "Загрузка данных",
    "left.label.dataset_dir": "Папка с изображениями",
    "left.label.video_path": "Видео (MP4/AVI/MOV)",
    "left.label.frame_stride": "Шаг",
    "left.label.max_frames": "Кол-во кадров",
    "left.label.model_path": "Путь к модели сегментации",
    "left.label.sam2_config": "YAML-конфиг SAM 2",
    "left.label.engine": "Модель сегментатора",
    "left.label.sam_model_type": "Тип модели SAM (vit_h / vit_l / vit_b)",
    "left.advanced.summary": "Автосегментация (настройки гиперпараметров SAM)",
    "left.advanced.points_per_side": "Точек на сторону",
    "left.advanced.points_per_batch": "Точек в батче",
    "left.advanced.pred_iou_thresh": "Порог IoU",
    "left.advanced.stability_thresh": "Порог устойчивости",
    "left.advanced.stability_offset": "Смещение устойчивости",
    "left.advanced.box_nms": "NMS боксов",
    "left.advanced.crop_layers": "Слои кропа",
    "left.advanced.crop_nms": "NMS кропа",
    "left.advanced.crop_overlap": "Перекрытие кропа",
    "left.advanced.crop_points_scale": "Масштаб точек кропа",
    "left.advanced.merge_same_class": "Объединять маски одного класса",
    "left.label.labels_dir": "Папка для меток (опционально)",
    "left.label.classes_file": "Файл классов YAML/TXT (опционально)",
    "left.label.classes": "Классы (по одному в строке)",
    "left.button.load": "Загрузить",
    "left.video_info.not_loaded": "Видео не загружено.",
    "left.video_info.details": "Кадры: {frameCount} | FPS: {fps} | {width}x{height} | шаг {stride}",
    "left.hint.tip":
      "Подсказка: укажите либо папку с изображениями, либо путь к видео и нажмите «Загрузить».",
    "left.controls.title": "Управление:",
    "left.controls.select": "ЛКМ: выбрать маску",
    "left.controls.negative_point": "Shift/Alt + ЛКМ: отрицательная точка",
    "left.controls.pan": "Колесо мыши: перемещение по изображению",
    "left.controls.add_vertex": "Добавить вершину: <code>Ctrl</code> + клик по ребру",
    "left.controls.draw": "Рисование: <code>N</code>, затем точки, <code>Enter</code> — завершить",
    "left.controls.delete": "Удаление: <code>Del</code> или кнопка «Удалить маску»",
    "main.nav.prev": "Назад",
    "main.nav.next": "Вперед",
    "main.nav.next_unlabeled": "Кадр без меток",
    "main.actions.auto_segment": "Автосегментация",
    "main.actions.save": "Сохранить",
    "main.actions.save_all": "Сохранить все",
    "main.label.conf": "Conf(YOLO)",
    "main.label.min_area": "Мин. площадь",
    "main.zoom.fit": "По умолчанию",
    "status.ready": "Готово.",
    "right.title": "Маски",
    "right.label.class": "Класс для сегментации/рисования",
    "right.section.points": "Точки (SAM2)",
    "right.button.point_mode": "Режим точек",
    "right.button.clear_points": "Очистить точки",
    "right.button.segment_points": "Сегментация по точкам",
    "right.button.propagate_video": "Распространить по видео",
    "right.checkbox.refine": "Уточнять выбранную маску",
    "right.checkbox.merge_contours": "Объединять контуры в одну маску",
    "right.checkbox.seed_mask": "Якорная маска",
    "right.checkbox.copy_prev": "Копировать маску далее",
    "right.hint.negative_point": "Shift/Alt: отрицательная точка",
    "right.hint.enter_backspace":
      "<code>Enter</code> &mdash; сегментировать, <code>Backspace</code> &mdash; удалить последнюю точку",
    "right.button.delete_mask": "Удалить маску",
    "right.button.clear_masks": "Очистить маски",
    "right.box_list_title": "Маски",
    "right.button.remove_unchecked": "Удалить исключённые",
    "right.button.check_all": "Отметить все",
    "right.hotkeys.title": "Горячие клавиши",
    "right.hotkeys.save": "<code>Ctrl+S</code> &mdash; сохранить",
    "right.hotkeys.save_all": "<code>Ctrl+Shift+S</code> &mdash; сохранить все",
    "right.hotkeys.auto": "<code>P</code> &mdash; автосегментация",
    "right.hotkeys.draw": "<code>N</code> &mdash; рисование маски",
    "right.hotkeys.point_mode": "<code>T</code> &mdash; режим точек",
    "right.hotkeys.insert_vertex": "<code>Ctrl</code> + клик по ребру &mdash; добавить вершину",
    "right.hotkeys.nav": "<code>Left/Right</code> &mdash; назад/вперед",
    "right.hotkeys.undo": "<code>Ctrl+Z</code> &mdash; отмена",
    "right.hotkeys.zoom": "<code>+ / -</code> &mdash; масштаб",
    "right.hotkeys.fit": "<code>0</code> &mdash; по экрану",
    "picker.title": "Выберите путь",
    "picker.close": "Закрыть",
    "picker.roots": "Корень",
    "picker.up": "Вверх",
    "picker.select_current": "Выбрать текущую папку",
    "picker.select_dataset": "Выберите папку с изображениями",
    "picker.select_model": "Выберите файл модели",
    "picker.select_sam2_yaml": "Выберите YAML-конфиг SAM 2",
    "picker.select_labels": "Выберите папку для меток",
    "picker.select_classes": "Выберите файл классов YAML/TXT",
    "picker.select_video": "Выберите видеофайл",
    "picker.tag.dir": "ПАПКА",
    "picker.tag.file": "ФАЙЛ",
    "picker.empty": "Нет файлов",
    "points.mode": "Режим точек",
    "points.mode_on": "Режим точек: ВКЛ",
    "segment.area": "Площадь {area} px",
    "status.initial": "Укажите папку с изображениями или путь к видео и нажмите «Загрузить».",
    "status.current":
      "{path} | {index}/{total} | масок: {count} (оставлено {kept}){dirty} | несохранённых: {unsaved}{tail}",
    "status.dirty_suffix": " (не сохранено)",
    "status.extra.cached": "из кэша",
    "status.extra.copied_prev": "скопировано с предыдущего кадра",
    "status.extra.loaded": "загружено",
    "status.extra.saved_to": "Сохранено в {path}",
    "status.extra.saved_counts": "Сохранено: {saved}, ошибок {errors}",
    "status.point_mode_cleared": "Режим точек: точки очищены.",
    "status.point_mode_help": "Режим точек: ЛКМ добавляет, ПКМ удаляет.",
    "status.vertex_inserted": "Добавлена вершина.",
    "status.draw_mode_help": "Режим рисования: кликайте, чтобы ставить точки; двойной клик или Enter — завершить.",
    "status.undo_applied": "Отмена выполнена.",
    "status.undo_none": "Отменять нечего.",
    "status.failed_load_roots": "Не удалось загрузить корни: {error}",
    "status.failed_list_path": "Не удалось открыть папку: {error}",
    "status.no_images_found": "В папке нет изображений.",
    "status.both_dataset_and_video": "Указаны и папка, и видео. Загружаю видео.",
    "status.dataset_or_video_required": "Нужно указать папку с изображениями или путь к видео.",
    "status.dataset_required": "Нужно указать папку с изображениями.",
    "status.loading_dataset": "Загрузка изображений...",
    "status.failed_load_dataset": "Не удалось загрузить изображения: {error}",
    "status.video_path_required": "Нужно указать путь к видео.",
    "status.extracting_frames": "Извлекаю кадры из видео...",
    "status.video_loaded": "Видео загружено. Размечайте кадры как изображения.",
    "status.failed_load_video": "Не удалось загрузить видео: {error}",
    "status.loading_image": "Загрузка изображения...",
    "status.failed_load_image": "Не удалось загрузить изображение: {error}",
    "status.no_unlabeled_found": "Нет изображений без разметки.",
    "status.no_image_selected": "Изображение не выбрано.",
    "status.running_auto_seg": "Автосегментация...",
    "status.segmentation_complete": "Сегментация завершена: масок {count}. Проверьте список.",
    "status.segmentation_failed": "Ошибка сегментации: {error}",
    "status.add_points_first": "Добавьте точки.",
    "status.need_positive_point": "Нужна хотя бы одна положительная точка.",
    "status.sam2_required_select": "Нужен SAM 2. Выберите его в настройках сегментации.",
    "status.sam2_yaml_required": "Нужен YAML-конфиг SAM 2.",
    "status.sam2_model_required": "Нужен путь к модели SAM 2.",
    "status.segmenting_points": "Сегментация по точкам...",
    "status.no_masks_returned": "Маски не найдены.",
    "status.failed_choose_primary": "Не удалось выбрать основную маску для уточнения.",
    "status.mask_updated_points": "Маска обновлена по точкам. Можно добавить ещё точки.",
    "status.added_masks_points": "Добавлено масок по точкам: {count}.",
    "status.point_seg_failed": "Ошибка сегментации по точкам: {error}",
    "status.load_video_first": "Сначала загрузите видео.",
    "status.sam2_required": "Нужен SAM 2.",
    "status.sam2_yaml_required_prop": "Нужен YAML-конфиг SAM 2 для распространения.",
    "status.sam2_model_required_prop": "Нужен путь к модели SAM 2 для распространения.",
    "status.propagating": "Распространяю маску по видео... это может занять время.",
    "status.propagation_complete": "Распространение завершено: сохранено {saved}, ошибок {errors}.",
    "status.propagation_failed": "Ошибка распространения: {error}",
    "status.saving": "Сохранение...",
    "status.save_failed": "Ошибка сохранения: {error}",
    "status.nothing_to_save": "Нет изменений для сохранения.",
    "status.batch_saving": "Пакетное сохранение: {count}...",
    "status.batch_save_failed": "Ошибка пакетного сохранения: {error}",
    "status.removed_unchecked_masks": "Удалено исключённых масок: {count}.",
    "status.no_mask_selected_delete": "Не выбрана маска для удаления.",
    "error.invalid_response": "Некорректный ответ: {text}",
    "error.request_failed": "Ошибка запроса ({status})",
    "error.failed_load_image": "Не удалось загрузить изображение.",
  },
};

function formatTemplate(template, params) {
  if (!params) {
    return template;
  }
  return template.replace(/\{(\w+)\}/g, (match, key) => {
    if (Object.prototype.hasOwnProperty.call(params, key)) {
      return String(params[key]);
    }
    return match;
  });
}

function tr(key, params = null) {
  const lang = state.lang || "en";
  const table = I18N[lang] || I18N.en;
  const template = (table && table[key]) || (I18N.en && I18N.en[key]) || key;
  return formatTemplate(template, params);
}

function applyI18nToDom() {
  document.documentElement.lang = state.lang || "en";
  document.title = tr("app.title");

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = tr(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = tr(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-html]").forEach((node) => {
    node.innerHTML = tr(node.dataset.i18nHtml);
  });
}

function setLanguage(lang, persist = true) {
  const next = lang === "ru" ? "ru" : "en";
  state.lang = next;
  if (el.languageSelect) {
    el.languageSelect.value = next;
  }
  if (persist) {
    try {
      localStorage.setItem(LANG_STORAGE_KEY, next);
    } catch (_) {
      // ignore storage errors
    }
  }
  applyI18nToDom();
  updatePointButtons();
  updateVideoInfo();
  refreshSegmentList();
  if (state.picker.target && state.picker.lastData && el.pickerModal && !el.pickerModal.classList.contains("hidden")) {
    renderPickerList(state.picker.lastData);
  }
  if (state.imagePath) {
    updateStatusForCurrent();
  } else {
    setStatus(tr("status.initial"));
  }
}

function initLanguage() {
  let lang = null;
  try {
    lang = localStorage.getItem(LANG_STORAGE_KEY);
  } catch (_) {
    lang = null;
  }
  if (!lang) {
    const nav = (navigator.languages && navigator.languages[0]) || navigator.language || "";
    lang = nav.toLowerCase().startsWith("ru") ? "ru" : "en";
  }
  setLanguage(lang, false);
  if (el.languageSelect) {
    el.languageSelect.addEventListener("change", () => setLanguage(el.languageSelect.value, true));
  }
}

function setStatus(message, isError = false) {
  el.statusBar.textContent = message;
  el.statusBar.classList.toggle("error", isError);
}

function updateVideoInfo() {
  if (!el.videoInfo) {
    return;
  }
  if (!state.video.active || !state.video.info) {
    el.videoInfo.textContent = tr("left.video_info.not_loaded");
    return;
  }
  const info = state.video.info;
  const fpsValue = Number.isFinite(info.fps) ? info.fps.toFixed(2) : "0.00";
  const frameCount = Number.isFinite(info.frame_count) ? info.frame_count : 0;
  el.videoInfo.textContent = tr("left.video_info.details", {
    frameCount,
    fps: fpsValue,
    width: info.width,
    height: info.height,
    stride: info.stride,
  });
}

function setMode(mode, videoInfo = null) {
  state.mode = mode;
  state.video.active = mode === "video";
  state.video.info = videoInfo;
  if (el.propagateVideoBtn) {
    el.propagateVideoBtn.style.display = state.video.active ? "inline-flex" : "none";
  }
  updateVideoInfo();
}

function updatePointButtons() {
  if (!el.pointModeBtn) {
    return;
  }
  el.pointModeBtn.classList.toggle("btn-primary", state.points.active);
  el.pointModeBtn.textContent = state.points.active ? tr("points.mode_on") : tr("points.mode");
}

function clearPoints(silent = false) {
  state.points.items = [];
  render();
  if (!silent && state.points.active) {
    setStatus(tr("status.point_mode_cleared"));
  }
}

function removeLastPoint() {
  if (!state.points.items.length) {
    return;
  }
  state.points.items.pop();
  render();
}

function togglePointMode() {
  const next = !state.points.active;
  state.points.active = next;
  if (next) {
    state.draw.active = false;
    state.draw.points = [];
    state.draw.preview = null;
  } else {
    state.points.items = [];
  }
  refreshCanvasCursor();
  updatePointButtons();
  if (next) {
    setStatus(tr("status.point_mode_help"));
  } else {
    updateStatusForCurrent();
  }
  render();
}

function renderPoints() {
  if (!state.points.items.length) {
    return;
  }
  state.points.items.forEach((point) => {
    const canvasPoint = imageToCanvas(point);
    ctx.beginPath();
    ctx.arc(canvasPoint.x, canvasPoint.y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 1 ? "rgba(34, 197, 94, 0.9)" : "rgba(239, 68, 68, 0.9)";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.2;
    ctx.fill();
    ctx.stroke();
  });
}

function parseClassesFromInput() {
  return el.classesInput.value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function deepCopySegments(segments) {
  return segments.map((segment) => ({
    class_id: segment.class_id,
    points: segment.points.map((point) => ({ x: point.x, y: point.y })),
    score: typeof segment.score === "number" ? segment.score : undefined,
    keep: segment.keep !== false,
  }));
}

function normalizeSegments(segments) {
  if (!Array.isArray(segments)) {
    return [];
  }
  return segments
    .map((segment) => {
      const classId = Number.isFinite(segment.class_id) ? Math.max(0, segment.class_id) : 0;
      const rawPoints = Array.isArray(segment.points) ? segment.points : [];
      const points = rawPoints
        .map((point) => {
          if (Array.isArray(point) && point.length >= 2) {
            return { x: Number(point[0]) || 0, y: Number(point[1]) || 0 };
          }
          if (point && typeof point === "object") {
            return { x: Number(point.x) || 0, y: Number(point.y) || 0 };
          }
          return null;
        })
        .filter(Boolean);
      return {
        class_id: classId,
        points,
        score: Number.isFinite(segment.score) ? segment.score : undefined,
        keep: segment.keep !== false,
      };
    })
    .filter((segment) => segment.points.length >= 3);
}

function getHistoryForPath(path) {
  if (!state.historyByPath[path]) {
    state.historyByPath[path] = [];
  }
  return state.historyByPath[path];
}

function pushHistoryForCurrent() {
  if (!state.imagePath) {
    return;
  }
  const history = getHistoryForPath(state.imagePath);
  history.push({
    segments: deepCopySegments(state.segments),
    selectedIndex: state.selectedIndex,
  });
  if (history.length > 80) {
    history.shift();
  }
}

function undoCurrent() {
  if (!state.imagePath) {
    return false;
  }
  const history = getHistoryForPath(state.imagePath);
  if (history.length === 0) {
    return false;
  }
  const snapshot = history.pop();
  state.segments = deepCopySegments(snapshot.segments);
  state.selectedIndex = snapshot.selectedIndex;
  refreshSegmentList();
  render();
  updateCacheForCurrent(true);
  return true;
}

function setCurrentDirty(value) {
  if (!state.imagePath) {
    return;
  }
  if (value) {
    state.dirtyPaths.add(state.imagePath);
  } else {
    state.dirtyPaths.delete(state.imagePath);
  }
  state.dirty = state.dirtyPaths.has(state.imagePath);
  updateStatusForCurrent();
}

function updateCacheForCurrent(isDirty = true) {
  if (!state.imagePath) {
    return;
  }
  state.annotationsByPath[state.imagePath] = deepCopySegments(state.segments);
  state.annotationCountByPath[state.imagePath] = state.segments.length;
  setCurrentDirty(isDirty);
}

function onCanvasMouseDown(event) {
  if (!state.image) {
    return;
  }
  if (el.canvas && typeof el.canvas.focus === "function") {
    el.canvas.focus();
  }
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;

  if (event.button === 1) {
    event.preventDefault();
    state.interaction = {
      type: "pan",
      startCanvas: pointerCanvas,
      originOffsetX: state.view.offsetX,
      originOffsetY: state.view.offsetY,
    };
    render();
    return;
  }
  if (event.button !== 0) {
    if (state.points.active && event.button === 2) {
      event.preventDefault();
      const imagePoint = clampPointToImage(canvasToImage(pointerCanvas));
      state.points.items.push({ x: imagePoint.x, y: imagePoint.y, label: 0 });
      render();
    }
    return;
  }

  if (state.points.active) {
    event.preventDefault();
    const imagePoint = clampPointToImage(canvasToImage(pointerCanvas));
    const isNegative = event.shiftKey || event.altKey;
    state.points.items.push({ x: imagePoint.x, y: imagePoint.y, label: isNegative ? 0 : 1 });
    render();
    return;
  }

  if (state.draw.active) {
    const point = clampPointToImage(canvasToImage(pointerCanvas));
    state.draw.points.push(point);
    state.draw.preview = point;
    render();
    return;
  }

  const imagePoint = clampPointToImage(canvasToImage(pointerCanvas));

  if (event.ctrlKey) {
    if (state.selectedIndex >= 0) {
      const selected = state.segments[state.selectedIndex];
      const edgeIndex = hitEdge(pointerCanvas, selected);
      if (edgeIndex >= 0) {
        pushHistoryForCurrent();
        insertVertex(state.selectedIndex, edgeIndex, imagePoint);
        setStatus(tr("status.vertex_inserted"));
        return;
      }
    }
    for (let i = state.segments.length - 1; i >= 0; i -= 1) {
      const edgeIndex = hitEdge(pointerCanvas, state.segments[i]);
      if (edgeIndex >= 0) {
        pushHistoryForCurrent();
        insertVertex(i, edgeIndex, imagePoint);
        setStatus(tr("status.vertex_inserted"));
        return;
      }
    }
  }

  if (state.selectedIndex >= 0) {
    const selected = state.segments[state.selectedIndex];
    const vertexIndex = hitVertex(pointerCanvas, selected);
    if (vertexIndex >= 0) {
      pushHistoryForCurrent();
      state.interaction = {
        type: "vertex",
        segmentIndex: state.selectedIndex,
        vertexIndex,
      };
      return;
    }
  }

  const segmentIndex = hitSegment(imagePoint);
  if (segmentIndex >= 0) {
    selectSegment(segmentIndex);
    pushHistoryForCurrent();
    state.interaction = {
      type: "move",
      segmentIndex,
      startPoint: imagePoint,
      originPoints: state.segments[segmentIndex].points.map((point) => ({ x: point.x, y: point.y })),
    };
    return;
  }

  state.selectedIndex = -1;
  refreshSegmentList();
  render();
}

function onCanvasMouseMove(event) {
  if (!state.image) {
    return;
  }
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;
  if (state.draw.active) {
    state.draw.preview = clampPointToImage(canvasToImage(pointerCanvas));
    render();
  }
  if (!state.interaction) {
    return;
  }

  if (state.interaction.type === "pan") {
    const pan = state.interaction;
    const dx = pointerCanvas.x - pan.startCanvas.x;
    const dy = pointerCanvas.y - pan.startCanvas.y;
    state.view.offsetX = pan.originOffsetX + dx;
    state.view.offsetY = pan.originOffsetY + dy;
    clampViewOffset();
    render();
    return;
  }

  const pointer = clampPointToImage(canvasToImage(pointerCanvas));

  if (state.interaction.type === "move") {
    const move = state.interaction;
    const dx = pointer.x - move.startPoint.x;
    const dy = pointer.y - move.startPoint.y;
    const updated = move.originPoints.map((point) => ({ x: point.x + dx, y: point.y + dy }));
    state.segments[move.segmentIndex].points = updated.map(clampPointToImage);
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
    return;
  }

  if (state.interaction.type === "vertex") {
    const vertex = state.interaction;
    const segment = state.segments[vertex.segmentIndex];
    segment.points[vertex.vertexIndex] = pointer;
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
  }
}

function onCanvasMouseUp() {
  if (state.interaction) {
    state.interaction = null;
    refreshCanvasCursor();
  }
}

function onCanvasWheel(event) {
  if (!state.image) {
    return;
  }
  event.preventDefault();
  const pointer = getPointer(event);
  state.view.lastPointer = pointer;
  zoomBy(event.deltaY < 0 ? 1.1 : 0.9, pointer);
}

function onCanvasDoubleClick(event) {
  if (!state.draw.active) {
    return;
  }
  event.preventDefault();
  finalizeDraw();
}

function onBoxListClick(event) {
  const item = event.target.closest("li");
  if (!item) {
    return;
  }
  const index = Number(item.dataset.index);
  if (!Number.isNaN(index)) {
    selectSegment(index);
  }
}

function toggleDrawMode() {
  state.draw.active = !state.draw.active;
  state.draw.points = [];
  state.draw.preview = null;
  if (state.draw.active) {
    state.points.active = false;
    state.points.items = [];
    updatePointButtons();
  }
  refreshCanvasCursor();
  render();
  if (state.draw.active) {
    setStatus(tr("status.draw_mode_help"));
  } else {
    updateStatusForCurrent();
  }
}

function onKeyDown(event) {
  const key = event.key;
  const lower = key.toLowerCase();

  if (key === "Escape") {
    if (state.draw.active) {
      event.preventDefault();
      cancelDraw();
      updateStatusForCurrent();
      return;
    }
    if (state.points.active) {
      event.preventDefault();
      clearPoints();
      updateStatusForCurrent();
      return;
    }
  }

  if (event.ctrlKey && lower === "s" && event.shiftKey) {
    event.preventDefault();
    saveAllDirty();
    return;
  }
  if (event.ctrlKey && lower === "s") {
    event.preventDefault();
    saveCurrentImage();
    return;
  }
  if (event.ctrlKey && lower === "z") {
    if (!canUseGlobalHotkeys()) {
      return;
    }
    event.preventDefault();
    if (undoCurrent()) {
      setStatus(tr("status.undo_applied"));
    } else {
      setStatus(tr("status.undo_none"));
    }
    return;
  }

  if (!canUseGlobalHotkeys()) {
    return;
  }

  if (state.points.active && key === "Backspace") {
    event.preventDefault();
    removeLastPoint();
    return;
  }

  if (lower === "p") {
    event.preventDefault();
    predictCurrentImage();
    return;
  }
  if (lower === "n") {
    event.preventDefault();
    toggleDrawMode();
    return;
  }
  if (lower === "t") {
    event.preventDefault();
    togglePointMode();
    return;
  }
  if (key === "Enter" && state.draw.active) {
    event.preventDefault();
    finalizeDraw();
    return;
  }
  if (key === "Enter" && state.points.active) {
    event.preventDefault();
    segmentFromPoints();
    return;
  }
  if (key === "Delete" || key === "Backspace") {
    event.preventDefault();
    deleteSelectedSegment();
    return;
  }
  if (key === "+" || key === "=") {
    event.preventDefault();
    zoomBy(1.12);
    return;
  }
  if (key === "-" || key === "_") {
    event.preventDefault();
    zoomBy(0.88);
    return;
  }
  if (key === "0") {
    event.preventDefault();
    zoomFit();
    return;
  }

  if (key === "ArrowLeft") {
    event.preventDefault();
    navigateByStep(-1);
    return;
  }
  if (key === "ArrowRight") {
    event.preventDefault();
    navigateByStep(1);
  }
}

function setPickerVisible(visible) {
  el.pickerModal.classList.toggle("hidden", !visible);
}

function selectPickerPath(path) {
  if (!state.picker.target) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  config.input.value = path;
  closePicker();
}

function renderPickerList(data) {
  state.picker.lastData = data;
  el.pickerList.innerHTML = "";
  el.pickerPath.textContent = data.current_path || tr("picker.roots");
  state.picker.currentPath = data.current_path || null;
  state.picker.parentPath = data.parent_path || null;

  if (!state.picker.target) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  el.pickerSelectCurrentBtn.style.display = config.allowFolder ? "inline-block" : "none";
  el.pickerUpBtn.disabled = !data.parent_path;

  data.directories.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => loadPickerPath(entry.path, config.mode));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = tr("picker.tag.dir");
    const name = document.createElement("span");
    name.className = "name";
    name.appendChild(button);
    li.appendChild(name);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  data.files.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => selectPickerPath(entry.path));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = tr("picker.tag.file");
    const name = document.createElement("span");
    name.className = "name";
    name.appendChild(button);
    li.appendChild(name);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  if (data.directories.length === 0 && data.files.length === 0) {
    const empty = document.createElement("li");
    empty.className = "picker-item";
    empty.innerHTML = `<span class="name">${tr("picker.empty")}</span>`;
    el.pickerList.appendChild(empty);
  }
}

async function showPickerRoots() {
  try {
    const data = await apiJson("/api/fs/roots");
    renderPickerList({
      current_path: "",
      parent_path: null,
      directories: (data.roots || []).map((path) => ({ name: path, path })),
      files: [],
    });
  } catch (error) {
    setStatus(tr("status.failed_load_roots", { error: error.message }), true);
  }
}

async function loadPickerPath(path, mode) {
  try {
    const data = await apiJson(`/api/fs/list?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}`);
    renderPickerList(data);
  } catch (error) {
    setStatus(tr("status.failed_list_path", { error: error.message }), true);
  }
}

function closePicker() {
  state.picker.target = null;
  state.picker.currentPath = null;
  state.picker.parentPath = null;
  state.picker.lastData = null;
  setPickerVisible(false);
}

function openPicker(target) {
  const config = PICKER_CONFIG[target];
  if (!config) {
    return;
  }
  state.picker.target = target;
  el.pickerTitle.textContent = tr(config.titleKey);
  setPickerVisible(true);

  const startPath = config.input.value.trim() || el.datasetDirInput.value.trim();
  if (startPath) {
    loadPickerPath(startPath, config.mode);
  } else {
    showPickerRoots();
  }
}

function onPickerSelectCurrent() {
  if (!state.picker.target || !state.picker.currentPath) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  if (!config.allowFolder) {
    return;
  }
  config.input.value = state.picker.currentPath;
  closePicker();
}

function onPickerUp() {
  if (!state.picker.target || !state.picker.parentPath) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  loadPickerPath(state.picker.parentPath, config.mode);
}

el.loadSessionBtn.addEventListener("click", loadData);
el.predictBtn.addEventListener("click", predictCurrentImage);
el.saveBtn.addEventListener("click", saveCurrentImage);
el.saveAllBtn.addEventListener("click", saveAllDirty);
el.deleteBtn.addEventListener("click", deleteSelectedSegment);
el.clearBtn.addEventListener("click", clearSegments);
if (el.pointModeBtn) {
  el.pointModeBtn.addEventListener("click", togglePointMode);
}
if (el.clearPointsBtn) {
  el.clearPointsBtn.addEventListener("click", clearPoints);
}
if (el.segmentPointsBtn) {
  el.segmentPointsBtn.addEventListener("click", segmentFromPoints);
}
if (el.propagateVideoBtn) {
  el.propagateVideoBtn.addEventListener("click", propagateVideo);
}
el.prevBtn.addEventListener("click", () => navigateByStep(-1));
el.nextBtn.addEventListener("click", () => navigateByStep(1));
if (el.nextUnlabeledBtn) {
  el.nextUnlabeledBtn.addEventListener("click", jumpToNextUnlabeled);
}
el.zoomInBtn.addEventListener("click", () => zoomBy(1.12));
el.zoomOutBtn.addEventListener("click", () => zoomBy(0.88));
el.zoomFitBtn.addEventListener("click", zoomFit);
el.removeUncheckedBtn.addEventListener("click", removeUncheckedSegments);
el.checkAllBtn.addEventListener("click", checkAllSegments);

el.imageSelect.addEventListener("change", () => {
  const index = Number(el.imageSelect.value);
  if (!Number.isNaN(index)) {
    loadImageByIndex(index);
  }
});

el.classSelect.addEventListener("change", onClassChanged);
el.boxList.addEventListener("click", onBoxListClick);

el.canvas.addEventListener("mousedown", onCanvasMouseDown);
el.canvas.addEventListener("mousemove", onCanvasMouseMove);
el.canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
el.canvas.addEventListener("dblclick", onCanvasDoubleClick);
el.canvas.addEventListener("contextmenu", (event) => {
  if (state.points.active) {
    event.preventDefault();
  }
});
window.addEventListener("mouseup", onCanvasMouseUp);
window.addEventListener("keydown", onKeyDown);
window.addEventListener("resize", resizeCanvas);

el.browseDatasetBtn.addEventListener("click", () => openPicker("dataset"));
el.browseModelBtn.addEventListener("click", () => openPicker("model"));
el.browseSam2ConfigBtn.addEventListener("click", () => openPicker("sam2Config"));
el.browseLabelsBtn.addEventListener("click", () => openPicker("labels"));
el.browseClassesBtn.addEventListener("click", () => openPicker("classes"));
if (el.browseVideoBtn) {
  el.browseVideoBtn.addEventListener("click", () => openPicker("video"));
}
el.pickerCloseBtn.addEventListener("click", closePicker);
el.pickerRootsBtn.addEventListener("click", showPickerRoots);
el.pickerSelectCurrentBtn.addEventListener("click", onPickerSelectCurrent);
el.pickerUpBtn.addEventListener("click", onPickerUp);
el.pickerModal.addEventListener("click", (event) => {
  if (event.target === el.pickerModal) {
    closePicker();
  }
});

resizeCanvas();
updateZoomLabel();
syncClassControls();
refreshSegmentList();
initLanguage();
updatePointButtons();
setMode("image");
function refreshSegmentList() {
  el.boxList.innerHTML = "";
  state.segments.forEach((segment, index) => {
    const li = document.createElement("li");
    li.dataset.index = String(index);
    if (index === state.selectedIndex) {
      li.classList.add("active");
    }
    if (segment.keep === false) {
      li.classList.add("unchecked");
    }

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = segment.keep !== false;
    checkbox.addEventListener("change", (event) => {
      event.stopPropagation();
      segment.keep = checkbox.checked;
      updateCacheForCurrent(true);
      refreshSegmentList();
      render();
    });

    const meta = document.createElement("div");
    meta.className = "mask-meta";

    const title = document.createElement("div");
    title.className = "mask-title";
    title.textContent = `${index + 1}. ${classNameById(segment.class_id)}`;

    const subtitle = document.createElement("div");
    subtitle.className = "mask-subtitle";
    subtitle.textContent = tr("segment.area", { area: Math.round(polygonArea(segment.points)) });

    meta.appendChild(title);
    meta.appendChild(subtitle);

    li.appendChild(checkbox);
    li.appendChild(meta);
    el.boxList.appendChild(li);
  });
}

function drawSegment(segment, isSelected) {
  if (!segment.points || segment.points.length < 3) {
    return;
  }
  ctx.beginPath();
  segment.points.forEach((point, index) => {
    const canvasPoint = imageToCanvas(point);
    if (index === 0) {
      ctx.moveTo(canvasPoint.x, canvasPoint.y);
    } else {
      ctx.lineTo(canvasPoint.x, canvasPoint.y);
    }
  });
  ctx.closePath();
  ctx.fillStyle = segment.keep === false ? colorForClass(segment.class_id, 0.06) : colorForClass(segment.class_id, 0.22);
  ctx.strokeStyle = colorStrokeForClass(segment.class_id, isSelected ? 0.95 : 0.6);
  ctx.lineWidth = isSelected ? 2.4 : 1.2;
  ctx.fill();
  ctx.stroke();

  if (isSelected) {
    segment.points.forEach((point) => {
      const canvasPoint = imageToCanvas(point);
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = colorStrokeForClass(segment.class_id, 1);
      ctx.lineWidth = 1.2;
      ctx.fill();
      ctx.stroke();
    });
  }
}

function renderDrawPreview() {
  if (!state.draw.active || state.draw.points.length === 0) {
    return;
  }
  ctx.beginPath();
  state.draw.points.forEach((point, index) => {
    const canvasPoint = imageToCanvas(point);
    if (index === 0) {
      ctx.moveTo(canvasPoint.x, canvasPoint.y);
    } else {
      ctx.lineTo(canvasPoint.x, canvasPoint.y);
    }
  });
  if (state.draw.preview) {
    const previewCanvas = imageToCanvas(state.draw.preview);
    ctx.lineTo(previewCanvas.x, previewCanvas.y);
  }
  ctx.strokeStyle = "rgba(15, 118, 110, 0.9)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function render() {
  ctx.clearRect(0, 0, state.view.width, state.view.height);
  if (!state.image) {
    return;
  }
  ctx.drawImage(
    state.image,
    0,
    0,
    state.image.naturalWidth,
    state.image.naturalHeight,
    state.view.offsetX,
    state.view.offsetY,
    state.image.naturalWidth * state.view.scale,
    state.image.naturalHeight * state.view.scale
  );

  state.segments.forEach((segment, index) => {
    drawSegment(segment, index === state.selectedIndex);
  });

  renderDrawPreview();
  renderPoints();
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error(tr("error.invalid_response", { text }));
    }
  }
  if (!response.ok) {
    const detail =
      data && data.detail ? data.detail : tr("error.request_failed", { status: response.status });
    throw new Error(detail);
  }
  return data;
}

async function applySessionData(data, mode = "image") {
  state.sessionLoaded = true;
  state.images = data.images || [];
  state.imageIndex = state.images.length ? 0 : -1;
  state.imagePath = state.imageIndex >= 0 ? state.images[state.imageIndex] : "";
  state.classes = Array.isArray(data.classes) && data.classes.length ? data.classes : parseClassesFromInput();
  syncClassControls();
  syncImageSelect();
  state.annotationsByPath = {};
  state.annotationCountByPath = {};
  state.dirtyPaths.clear();
  state.selectedIndex = -1;
  state.segments = [];
  state.interaction = null;
  state.draw.active = false;
  state.draw.points = [];
  state.draw.preview = null;
  state.points.items = [];
  state.points.active = false;
  updatePointButtons();
  setMode(mode, data.video || null);
  if (state.images.length > 0) {
    await loadImageByIndex(state.imageIndex);
  } else {
    setStatus(tr("status.no_images_found"), true);
  }
}

async function loadData() {
  const videoPath = el.videoPathInput.value.trim();
  const datasetDir = el.datasetDirInput.value.trim();
  if (videoPath && datasetDir) {
    setStatus(tr("status.both_dataset_and_video"));
    return loadVideo();
  }
  if (videoPath) {
    return loadVideo();
  }
  if (datasetDir) {
    return loadSession();
  }
  setStatus(tr("status.dataset_or_video_required"), true);
}

async function loadSession() {
  const datasetDir = el.datasetDirInput.value.trim();
  if (!datasetDir) {
    setStatus(tr("status.dataset_required"), true);
    return;
  }
  const payload = {
    dataset_dir: datasetDir,
    classes: parseClassesFromInput(),
    classes_file: el.classesFileInput.value.trim() || null,
    model_path: el.modelPathInput.value.trim() || null,
    labels_dir: el.labelsDirInput.value.trim() || null,
  };
  setStatus(tr("status.loading_dataset"));
  try {
    const data = await apiJson("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await applySessionData(data, "image");
  } catch (error) {
    setStatus(tr("status.failed_load_dataset", { error: error.message }), true);
  }
}

async function loadVideo() {
  const videoPath = el.videoPathInput.value.trim();
  if (!videoPath) {
    setStatus(tr("status.video_path_required"), true);
    return;
  }
  const payload = {
    video_path: videoPath,
    frame_stride: Number(el.videoStrideInput.value) || 1,
    max_frames: el.videoMaxFramesInput.value ? Number(el.videoMaxFramesInput.value) : null,
    labels_dir: el.labelsDirInput.value.trim() || null,
    classes: parseClassesFromInput(),
    classes_file: el.classesFileInput.value.trim() || null,
  };
  setStatus(tr("status.extracting_frames"));
  try {
    const data = await apiJson("/api/video/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await applySessionData(data, "video");
    setStatus(tr("status.video_loaded"));
  } catch (error) {
    setStatus(tr("status.failed_load_video", { error: error.message }), true);
  }
}

async function loadAnnotations(path) {
  const data = await apiJson(`/api/annotations?path=${encodeURIComponent(path)}`);
  return normalizeSegments(data.segments || []);
}

async function loadImageByIndex(index) {
  if (index < 0 || index >= state.images.length) {
    return;
  }
  state.imageIndex = index;
  state.imagePath = state.images[index];
  state.draw.active = false;
  state.draw.points = [];
  state.draw.preview = null;
  state.points.items = [];
  refreshCanvasCursor();
  updateImageCounter();
  el.imageSelect.value = String(index);

  const cachedSegments = state.annotationsByPath[state.imagePath];
  if (cachedSegments) {
    state.segments = deepCopySegments(cachedSegments);
    state.selectedIndex = -1;
    refreshSegmentList();
    updateStatusForCurrent(tr("status.extra.cached"));
    render();
  } else {
    state.segments = [];
    refreshSegmentList();
  }

  const token = ++state.loadToken;
  setStatus(tr("status.loading_image"));
  try {
    const img = new Image();
    const imageUrl = `/api/image?path=${encodeURIComponent(state.imagePath)}`;
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error(tr("error.failed_load_image")));
      img.src = imageUrl;
    });
    if (token !== state.loadToken) {
      return;
    }
    state.image = img;
    fitImageInCanvas(state.view.userZoom);
    render();
      if (!cachedSegments) {
        state.segments = await loadAnnotations(state.imagePath);
        state.annotationsByPath[state.imagePath] = deepCopySegments(state.segments);
        state.annotationCountByPath[state.imagePath] = state.segments.length;
      }
      state.selectedIndex = -1;
      refreshSegmentList();
      if (
        state.mode === "video" &&
        el.carryPrevMaskInput &&
        el.carryPrevMaskInput.checked &&
        state.segments.length === 0 &&
        index > 0
      ) {
        const prevPath = state.images[index - 1];
        const prevSegments = state.annotationsByPath[prevPath];
        if (prevSegments && prevSegments.length) {
          state.segments = deepCopySegments(prevSegments);
          state.segments.forEach((segment) => {
            if (segment.keep === undefined) {
              segment.keep = true;
            }
          });
          updateCacheForCurrent(true);
          refreshSegmentList();
          render();
          updateStatusForCurrent(tr("status.extra.copied_prev"));
          return;
        }
      }
      updateStatusForCurrent(tr("status.extra.loaded"));
    } catch (error) {
      setStatus(tr("status.failed_load_image", { error: error.message }), true);
    }
  }

function navigateByStep(step) {
  if (state.images.length === 0) {
    return;
  }
  const nextIndex = clamp(state.imageIndex + step, 0, state.images.length - 1);
  if (nextIndex === state.imageIndex) {
    return;
  }
  loadImageByIndex(nextIndex);
}

function jumpToNextUnlabeled() {
  if (state.images.length === 0) {
    return;
  }
  for (let offset = 1; offset <= state.images.length; offset += 1) {
    const index = (state.imageIndex + offset) % state.images.length;
    const path = state.images[index];
    const count = state.annotationCountByPath[path] || 0;
    if (count === 0) {
      loadImageByIndex(index);
      return;
    }
  }
  setStatus(tr("status.no_unlabeled_found"));
}

function buildPredictPayload() {
  return {
    model_path: el.modelPathInput.value.trim() || null,
    sam2_config: el.sam2ConfigInput.value.trim() || null,
    engine: el.engineSelect.value || "sam",
    model_type: el.samModelTypeInput.value.trim() || null,
    conf: parseNumber(el.confInput.value, 0.25),
    points_per_side: parseNumber(el.pointsPerSideInput.value, 32),
    points_per_batch: parseNumber(el.pointsPerBatchInput.value, 64),
    pred_iou_thresh: parseNumber(el.predIouThreshInput.value, 0.8),
    stability_score_thresh: parseNumber(el.stabilityScoreThreshInput.value, 0.95),
    stability_score_offset: parseNumber(el.stabilityScoreOffsetInput.value, 1.0),
    box_nms_thresh: parseNumber(el.boxNmsThreshInput.value, 0.7),
    crop_n_layers: parseNumber(el.cropNLayersInput.value, 0),
    crop_nms_thresh: parseNumber(el.cropNmsThreshInput.value, 0.7),
    crop_overlap_ratio: parseNumber(el.cropOverlapRatioInput.value, 0.34),
    crop_n_points_downscale_factor: parseNumber(el.cropNPointsDownscaleInput.value, 1),
    min_area: parseNumber(el.minAreaInput.value, 0),
    merge_same_class: Boolean(el.mergeSameClassInput && el.mergeSameClassInput.checked),
  };
}

  function buildPointPayload() {
    return {
      model_path: el.modelPathInput.value.trim() || null,
      sam2_config: el.sam2ConfigInput.value.trim() || null,
      points: state.points.items.map((point) => ({ x: point.x, y: point.y, label: point.label })),
      multimask: state.points.items.length <= 1,
      min_area: Number(el.minAreaInput.value) || 0,
      simplify: 0.01,
      merge_contours: Boolean(el.mergeContoursInput && el.mergeContoursInput.checked),
    };
  }

async function predictCurrentImage() {
  if (!state.imagePath) {
    setStatus(tr("status.no_image_selected"), true);
    return;
  }
  const payload = buildPredictPayload();
  setStatus(tr("status.running_auto_seg"));
  try {
    const data = await apiJson(`/api/predict?path=${encodeURIComponent(state.imagePath)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (Array.isArray(data.classes) && data.classes.length > 0) {
      state.classes = data.classes;
      el.classesInput.value = state.classes.join("\n");
      syncClassControls();
    }
    if (state.classes.length === 0) {
      state.classes = ["class_0"];
      syncClassControls();
    }

    const predicted = normalizeSegments(data.segments || []);
    predicted.forEach((segment) => {
      segment.keep = false;
    });
    pushHistoryForCurrent();
    state.segments = predicted;
    state.selectedIndex = -1;
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
    setStatus(tr("status.segmentation_complete", { count: state.segments.length }));
  } catch (error) {
    setStatus(tr("status.segmentation_failed", { error: error.message }), true);
  }
}

async function segmentFromPoints() {
  if (!state.imagePath) {
    setStatus(tr("status.no_image_selected"), true);
    return;
  }
  if (!state.points.items.length) {
    setStatus(tr("status.add_points_first"), true);
    return;
  }
  if (!state.points.items.some((point) => point.label === 1)) {
    setStatus(tr("status.need_positive_point"), true);
    return;
  }
  if ((el.engineSelect.value || "sam") !== "sam2") {
    setStatus(tr("status.sam2_required_select"), true);
    return;
  }
  if (!el.sam2ConfigInput.value.trim()) {
    setStatus(tr("status.sam2_yaml_required"), true);
    return;
  }
  if (!el.modelPathInput.value.trim()) {
    setStatus(tr("status.sam2_model_required"), true);
    return;
  }

  const payload = buildPointPayload();
  const isVideo = state.mode === "video";
  const url = isVideo
    ? "/api/video/points"
    : `/api/predict_points?path=${encodeURIComponent(state.imagePath)}`;
  if (isVideo) {
    payload.frame_index = state.imageIndex;
    payload.clear_old_points = true;
  }

  setStatus(tr("status.segmenting_points"));
  try {
    const data = await apiJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const predicted = normalizeSegments(data.segments || []);
    if (!predicted.length) {
      setStatus(tr("status.no_masks_returned"), true);
      return;
    }
    const refineMode = Boolean(el.refinePointsInput && el.refinePointsInput.checked);
    if (refineMode) {
      const baseClassId =
        state.selectedIndex >= 0 && state.segments[state.selectedIndex]
          ? state.segments[state.selectedIndex].class_id
          : state.defaultClassId;
      const primary = pickLargestSegment(predicted);
      if (!primary) {
        setStatus(tr("status.failed_choose_primary"), true);
        return;
      }
      primary.keep = true;
      primary.class_id = baseClassId;
      pushHistoryForCurrent();
      if (state.selectedIndex >= 0 && state.segments[state.selectedIndex]) {
        state.segments[state.selectedIndex] = primary;
      } else {
        state.segments.push(primary);
        state.selectedIndex = state.segments.length - 1;
      }
      updateCacheForCurrent(true);
      refreshSegmentList();
      render();
      setStatus(tr("status.mask_updated_points"));
    } else {
      predicted.forEach((segment) => {
        segment.keep = true;
        segment.class_id = state.defaultClassId;
      });
      pushHistoryForCurrent();
      state.segments = state.segments.concat(predicted);
      state.selectedIndex = state.segments.length - 1;
      updateCacheForCurrent(true);
      refreshSegmentList();
      render();
      clearPoints(true);
      setStatus(tr("status.added_masks_points", { count: predicted.length }));
    }
  } catch (error) {
    setStatus(tr("status.point_seg_failed", { error: error.message }), true);
  }
}

async function propagateVideo() {
  if (state.mode !== "video") {
    setStatus(tr("status.load_video_first"), true);
    return;
  }
  if ((el.engineSelect.value || "sam") !== "sam2") {
    setStatus(tr("status.sam2_required"), true);
    return;
  }
  if (!el.sam2ConfigInput.value.trim()) {
    setStatus(tr("status.sam2_yaml_required_prop"), true);
    return;
  }
  if (!el.modelPathInput.value.trim()) {
    setStatus(tr("status.sam2_model_required_prop"), true);
    return;
  }
  const payload = {
    model_path: el.modelPathInput.value.trim() || null,
    sam2_config: el.sam2ConfigInput.value.trim() || null,
    start_frame: state.imageIndex >= 0 ? state.imageIndex : null,
    max_frames: null,
    reverse: false,
    min_area: Number(el.minAreaInput.value) || 0,
    simplify: 0.01,
    class_id: state.defaultClassId,
    merge_contours: Boolean(el.mergeContoursInput && el.mergeContoursInput.checked),
  };
  if (el.seedMaskForPropagateInput && el.seedMaskForPropagateInput.checked) {
    payload.seed_segments = toPayloadSegments(state.segments);
  }
  setStatus(tr("status.propagating"));
  try {
    const data = await apiJson("/api/video/propagate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.annotationsByPath = {};
    state.annotationCountByPath = {};
    if (state.imageIndex >= 0) {
      await loadImageByIndex(state.imageIndex);
    }
    setStatus(tr("status.propagation_complete", { saved: data.saved_count, errors: data.error_count }));
  } catch (error) {
    setStatus(tr("status.propagation_failed", { error: error.message }), true);
  }
}

function toPayloadSegments(segments) {
  return segments
    .filter((segment) => segment.keep !== false)
    .map((segment) => ({
      class_id: segment.class_id,
      points: segment.points.map((point) => [point.x, point.y]),
      score: typeof segment.score === "number" ? segment.score : undefined,
    }));
}

async function saveCurrentImage() {
  if (!state.imagePath) {
    setStatus(tr("status.no_image_selected"), true);
    return;
  }
  const keptSegments = state.segments.filter((segment) => segment.keep !== false);
  const payload = {
    segments: toPayloadSegments(state.segments),
  };
  setStatus(tr("status.saving"));
  try {
    const result = await apiJson(`/api/annotations?path=${encodeURIComponent(state.imagePath)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.segments = deepCopySegments(keptSegments);
    updateCacheForCurrent(false);
    refreshSegmentList();
    render();
    updateStatusForCurrent(tr("status.extra.saved_to", { path: result.label_path }));
  } catch (error) {
    setStatus(tr("status.save_failed", { error: error.message }), true);
  }
}

async function saveAllDirty() {
  const dirtyPaths = Array.from(state.dirtyPaths);
  if (dirtyPaths.length === 0) {
    setStatus(tr("status.nothing_to_save"));
    return;
  }
  const keptByPath = {};
  const items = dirtyPaths.map((path) => {
    const segments = state.annotationsByPath[path] || [];
    const kept = segments.filter((segment) => segment.keep !== false);
    keptByPath[path] = deepCopySegments(kept);
    return { path, segments: toPayloadSegments(segments) };
  });
  if (state.imagePath) {
    refreshSegmentList();
    render();
  }
  setStatus(tr("status.batch_saving", { count: items.length }));
  try {
    const result = await apiJson("/api/annotations/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });

    const failed = new Set((result.errors || []).map((item) => item.path));
    dirtyPaths.forEach((path) => {
      if (!failed.has(path)) {
        state.dirtyPaths.delete(path);
        state.annotationsByPath[path] = keptByPath[path] || [];
        state.annotationCountByPath[path] = (keptByPath[path] || []).length;
        if (path === state.imagePath) {
          state.segments = deepCopySegments(keptByPath[path] || []);
        }
      }
    });
    if (state.imagePath && !failed.has(state.imagePath)) {
      refreshSegmentList();
      render();
    }
    state.dirty = state.dirtyPaths.has(state.imagePath);
    updateStatusForCurrent(tr("status.extra.saved_counts", { saved: result.saved_count, errors: result.error_count }));
  } catch (error) {
    setStatus(tr("status.batch_save_failed", { error: error.message }), true);
  }
}

function removeUncheckedSegments() {
  const before = state.segments.length;
  state.segments = state.segments.filter((segment) => segment.keep !== false);
  if (state.segments.length !== before) {
    pushHistoryForCurrent();
    state.selectedIndex = Math.min(state.selectedIndex, state.segments.length - 1);
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
    setStatus(tr("status.removed_unchecked_masks", { count: before - state.segments.length }));
  }
}

function checkAllSegments() {
  if (!state.segments.length) {
    return;
  }
  state.segments.forEach((segment) => {
    segment.keep = true;
  });
  updateCacheForCurrent(true);
  refreshSegmentList();
  render();
}

function deleteSelectedSegment() {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.segments.length) {
    setStatus(tr("status.no_mask_selected_delete"));
    return;
  }
  pushHistoryForCurrent();
  state.segments.splice(state.selectedIndex, 1);
  state.selectedIndex = Math.min(state.selectedIndex, state.segments.length - 1);
  updateCacheForCurrent(true);
  refreshSegmentList();
  render();
}

function clearSegments() {
  if (!state.segments.length) {
    return;
  }
  pushHistoryForCurrent();
  state.segments = [];
  state.selectedIndex = -1;
  updateCacheForCurrent(true);
  refreshSegmentList();
  render();
}

function onClassChanged() {
  const classId = Number(el.classSelect.value);
  if (Number.isNaN(classId)) {
    return;
  }
  state.defaultClassId = classId;
  if (state.selectedIndex >= 0) {
    pushHistoryForCurrent();
    state.segments[state.selectedIndex].class_id = classId;
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
  }
}

function canUseGlobalHotkeys() {
  const active = document.activeElement;
  if (!active) {
    return true;
  }
  if (active.isContentEditable) {
    return false;
  }
  if (active.tagName === "TEXTAREA") {
    return false;
  }
  if (active.tagName === "INPUT") {
    const inputType = String(active.type || "").toLowerCase();
    return !["text", "search", "url", "email", "number", "password", "tel"].includes(inputType);
  }
  return true;
}

function finalizeDraw() {
  if (!state.draw.active) {
    return;
  }
  if (state.draw.points.length >= 3) {
    pushHistoryForCurrent();
    const segment = {
      class_id: state.defaultClassId,
      points: state.draw.points.map((point) => ({ x: point.x, y: point.y })),
      keep: true,
    };
    state.segments.push(segment);
    state.selectedIndex = state.segments.length - 1;
    updateCacheForCurrent(true);
    refreshSegmentList();
    render();
  }
  state.draw.active = false;
  state.draw.points = [];
  state.draw.preview = null;
  refreshCanvasCursor();
}

function cancelDraw() {
  if (!state.draw.active) {
    return;
  }
  state.draw.active = false;
  state.draw.points = [];
  state.draw.preview = null;
  refreshCanvasCursor();
  render();
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function parseNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function polygonArea(points) {
  if (!points || points.length < 3) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const next = points[(i + 1) % points.length];
    sum += points[i].x * next.y - next.x * points[i].y;
  }
  return Math.abs(sum) / 2;
}

function pickLargestSegment(segments) {
  if (!segments.length) {
    return null;
  }
  let best = segments[0];
  let bestArea = polygonArea(best.points);
  for (let i = 1; i < segments.length; i += 1) {
    const area = polygonArea(segments[i].points);
    if (area > bestArea) {
      best = segments[i];
      bestArea = area;
    }
  }
  return best;
}

function countKeptSegments(segments) {
  return segments.filter((segment) => segment.keep !== false).length;
}

function updateStatusForCurrent(extra = "") {
  if (!state.imagePath) {
    return;
  }
  const dirty = state.dirty ? tr("status.dirty_suffix") : "";
  const kept = countKeptSegments(state.segments);
  const tail = extra ? ` | ${extra}` : "";
  setStatus(
    tr("status.current", {
      path: state.imagePath,
      index: state.imageIndex + 1,
      total: state.images.length,
      count: state.segments.length,
      kept,
      dirty,
      unsaved: state.dirtyPaths.size,
      tail,
    })
  );
}

function colorForClass(classId, alpha = 0.18) {
  const hue = (classId * 57 + 29) % 360;
  return `hsla(${hue}, 78%, 46%, ${alpha})`;
}

function colorStrokeForClass(classId, alpha = 0.8) {
  const hue = (classId * 57 + 29) % 360;
  return `hsla(${hue}, 78%, 40%, ${alpha})`;
}

function clampPointToImage(point) {
  if (!state.image) {
    return point;
  }
  return {
    x: clamp(point.x, 0, state.image.naturalWidth),
    y: clamp(point.y, 0, state.image.naturalHeight),
  };
}

function refreshCanvasCursor() {
  if (!el.canvas) {
    return;
  }
  if (state.interaction && state.interaction.type === "pan") {
    el.canvas.style.cursor = "grabbing";
    return;
  }
  if (state.points.active) {
    el.canvas.style.cursor = "crosshair";
    return;
  }
  if (state.draw.active) {
    el.canvas.style.cursor = "crosshair";
    return;
  }
  el.canvas.style.cursor = "default";
}

function resizeCanvas() {
  const rect = el.canvasWrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));

  el.canvas.width = Math.floor(width * dpr);
  el.canvas.height = Math.floor(height * dpr);
  el.canvas.style.width = `${width}px`;
  el.canvas.style.height = `${height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.view.width = width;
  state.view.height = height;
  if (state.image) {
    fitImageInCanvas(state.view.userZoom);
  }
  render();
}

function clampViewOffset() {
  if (!state.image) {
    return;
  }
  const drawW = state.image.naturalWidth * state.view.scale;
  const drawH = state.image.naturalHeight * state.view.scale;
  if (drawW <= state.view.width) {
    state.view.offsetX = (state.view.width - drawW) / 2;
  } else {
    state.view.offsetX = clamp(state.view.offsetX, state.view.width - drawW, 0);
  }
  if (drawH <= state.view.height) {
    state.view.offsetY = (state.view.height - drawH) / 2;
  } else {
    state.view.offsetY = clamp(state.view.offsetY, state.view.height - drawH, 0);
  }
}

function applyScaleAroundPoint(userZoom, anchorCanvasPoint) {
  if (!state.image) {
    return;
  }
  const oldScale = state.view.scale || state.view.fitScale * state.view.userZoom || 1;
  const anchor = anchorCanvasPoint || { x: state.view.width / 2, y: state.view.height / 2 };
  const imagePoint = {
    x: (anchor.x - state.view.offsetX) / oldScale,
    y: (anchor.y - state.view.offsetY) / oldScale,
  };

  state.view.userZoom = clamp(userZoom, 0.1, 8);
  state.view.scale = state.view.fitScale * state.view.userZoom;
  state.view.offsetX = anchor.x - imagePoint.x * state.view.scale;
  state.view.offsetY = anchor.y - imagePoint.y * state.view.scale;
  clampViewOffset();
  updateZoomLabel();
}

function fitImageInCanvas(keepZoom = 1) {
  if (!state.image) {
    return;
  }
  const imageWidth = state.image.naturalWidth;
  const imageHeight = state.image.naturalHeight;
  state.view.fitScale = Math.min(state.view.width / imageWidth, state.view.height / imageHeight);
  state.view.userZoom = clamp(keepZoom, 0.1, 8);
  state.view.scale = state.view.fitScale * state.view.userZoom;
  state.view.offsetX = (state.view.width - imageWidth * state.view.scale) / 2;
  state.view.offsetY = (state.view.height - imageHeight * state.view.scale) / 2;
  clampViewOffset();
  updateZoomLabel();
}

function setZoom(zoom, anchorCanvasPoint = null) {
  if (!state.image) {
    state.view.userZoom = clamp(zoom, 0.1, 8);
    updateZoomLabel();
    return;
  }
  const anchor = anchorCanvasPoint || state.view.lastPointer || { x: state.view.width / 2, y: state.view.height / 2 };
  applyScaleAroundPoint(zoom, anchor);
  render();
}

function zoomBy(factor, anchorCanvasPoint = null) {
  setZoom(state.view.userZoom * factor, anchorCanvasPoint);
}

function zoomFit() {
  if (!state.image) {
    state.view.userZoom = 1;
    updateZoomLabel();
    return;
  }
  fitImageInCanvas(1);
  render();
}

function updateZoomLabel() {
  el.zoomLabel.textContent = `${Math.round(state.view.userZoom * 100)}%`;
}

function imageToCanvas(point) {
  return {
    x: point.x * state.view.scale + state.view.offsetX,
    y: point.y * state.view.scale + state.view.offsetY,
  };
}

function canvasToImage(point) {
  return {
    x: (point.x - state.view.offsetX) / state.view.scale,
    y: (point.y - state.view.offsetY) / state.view.scale,
  };
}

function getPointer(event) {
  const rect = el.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect =
      yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 0.000001) + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

function hitSegment(point) {
  for (let i = state.segments.length - 1; i >= 0; i -= 1) {
    if (pointInPolygon(point, state.segments[i].points)) {
      return i;
    }
  }
  return -1;
}

function hitVertex(pointerCanvas, segment) {
  const radius = 6;
  for (let i = 0; i < segment.points.length; i += 1) {
    const canvasPoint = imageToCanvas(segment.points[i]);
    const dx = canvasPoint.x - pointerCanvas.x;
    const dy = canvasPoint.y - pointerCanvas.y;
    if (dx * dx + dy * dy <= radius * radius) {
      return i;
    }
  }
  return -1;
}

function distanceToSegment(point, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = point.x - a.x;
  const apy = point.y - a.y;
  const abLenSq = abx * abx + aby * aby;
  if (abLenSq === 0) {
    const dx = point.x - a.x;
    const dy = point.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  let t = (apx * abx + apy * aby) / abLenSq;
  t = clamp(t, 0, 1);
  const projX = a.x + t * abx;
  const projY = a.y + t * aby;
  const dx = point.x - projX;
  const dy = point.y - projY;
  return Math.sqrt(dx * dx + dy * dy);
}

function hitEdge(pointerCanvas, segment, radius = 6) {
  const points = segment.points;
  if (!points || points.length < 2) {
    return -1;
  }
  let bestIndex = -1;
  let bestDist = radius;
  for (let i = 0; i < points.length; i += 1) {
    const a = imageToCanvas(points[i]);
    const b = imageToCanvas(points[(i + 1) % points.length]);
    const dist = distanceToSegment(pointerCanvas, a, b);
    if (dist <= bestDist) {
      bestDist = dist;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function insertVertex(segmentIndex, edgeIndex, imagePoint) {
  if (segmentIndex < 0 || edgeIndex < 0) {
    return false;
  }
  const segment = state.segments[segmentIndex];
  if (!segment || !segment.points || segment.points.length < 2) {
    return false;
  }
  const points = segment.points.slice();
  points.splice(edgeIndex + 1, 0, imagePoint);
  segment.points = points;
  state.selectedIndex = segmentIndex;
  updateCacheForCurrent(true);
  refreshSegmentList();
  render();
  return true;
}

function selectSegment(index) {
  state.selectedIndex = index;
  if (index >= 0 && index < state.segments.length) {
    const segment = state.segments[index];
    ensureClassCapacity(segment.class_id);
    el.classSelect.value = String(segment.class_id);
  }
  refreshSegmentList();
  render();
}

function ensureClassCapacity(maxClassId) {
  if (maxClassId < 0) {
    return;
  }
  while (state.classes.length <= maxClassId) {
    state.classes.push(`class_${state.classes.length}`);
  }
  if (state.classes.length === 0) {
    state.classes.push("class_0");
  }
}

function classNameById(classId) {
  return state.classes[classId] || `class_${classId}`;
}

function syncClassControls() {
  if (state.classes.length === 0) {
    state.classes = ["class_0"];
  }
  const prev = Number(el.classSelect.value);
  el.classSelect.innerHTML = "";
  state.classes.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index}: ${name}`;
    el.classSelect.appendChild(option);
  });
  if (Number.isFinite(prev) && prev >= 0 && prev < state.classes.length) {
    state.defaultClassId = prev;
  } else if (state.defaultClassId >= state.classes.length) {
    state.defaultClassId = 0;
  }
  el.classSelect.value = String(state.defaultClassId);
}

function syncImageSelect() {
  el.imageSelect.innerHTML = "";
  state.images.forEach((path, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = path;
    el.imageSelect.appendChild(option);
  });
  if (state.imageIndex >= 0 && state.imageIndex < state.images.length) {
    el.imageSelect.value = String(state.imageIndex);
  }
  updateImageCounter();
}

function updateImageCounter() {
  if (state.images.length === 0 || state.imageIndex < 0) {
    el.imageCounter.textContent = "0 / 0";
    return;
  }
  el.imageCounter.textContent = `${state.imageIndex + 1} / ${state.images.length}`;
}
