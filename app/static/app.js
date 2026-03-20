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
  dataset: { title: "Select dataset folder", mode: "dir", input: el.datasetDirInput, allowFolder: true },
  model: { title: "Select model file", mode: "model", input: el.modelPathInput, allowFile: true },
  sam2Config: { title: "Select SAM 2 YAML config", mode: "yaml", input: el.sam2ConfigInput, allowFile: true },
  labels: { title: "Select labels folder", mode: "dir", input: el.labelsDirInput, allowFolder: true },
  classes: { title: "Select classes YAML/TXT", mode: "yaml", input: el.classesFileInput, allowFile: true },
  video: { title: "Select video file", mode: "video", input: el.videoPathInput, allowFile: true },
};

const state = {
  sessionLoaded: false,
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
  picker: { target: null, currentPath: null, parentPath: null },
  draw: { active: false, points: [], preview: null },
  points: { active: false, items: [] },
};

function setStatus(message, isError = false) {
  el.statusBar.textContent = message;
  el.statusBar.classList.toggle("error", isError);
}

function updateVideoInfo() {
  if (!el.videoInfo) {
    return;
  }
  if (!state.video.active || !state.video.info) {
    el.videoInfo.textContent = "Video not loaded.";
    return;
  }
  const info = state.video.info;
  const fpsValue = Number.isFinite(info.fps) ? info.fps.toFixed(2) : "0.00";
  const frameCount = Number.isFinite(info.frame_count) ? info.frame_count : 0;
  el.videoInfo.textContent = `Frames: ${frameCount} | FPS: ${fpsValue} | ${info.width}x${info.height} | stride ${info.stride}`;
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
  el.pointModeBtn.textContent = state.points.active ? "Point mode: ON" : "Point mode";
}

function clearPoints(silent = false) {
  state.points.items = [];
  render();
  if (!silent && state.points.active) {
    setStatus("Point mode: points cleared.");
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
    setStatus("Point mode: LMB adds, RMB removes.");
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
        setStatus("Vertex inserted.");
        return;
      }
    }
    for (let i = state.segments.length - 1; i >= 0; i -= 1) {
      const edgeIndex = hitEdge(pointerCanvas, state.segments[i]);
      if (edgeIndex >= 0) {
        pushHistoryForCurrent();
        insertVertex(i, edgeIndex, imagePoint);
        setStatus("Vertex inserted.");
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
    setStatus("Draw mode: click to add points; double click or Enter to finish.");
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
      setStatus("Undo applied.");
    } else {
      setStatus("Nothing to undo.");
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
  el.pickerList.innerHTML = "";
  el.pickerPath.textContent = data.current_path || "Roots";
  state.picker.currentPath = data.current_path && data.current_path !== "Roots" ? data.current_path : null;
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
    tag.textContent = "DIR";
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
    tag.textContent = "FILE";
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
    empty.innerHTML = "<span class=\"name\">No files</span>";
    el.pickerList.appendChild(empty);
  }
}

async function showPickerRoots() {
  try {
    const data = await apiJson("/api/fs/roots");
    renderPickerList({
      current_path: "Roots",
      parent_path: null,
      directories: (data.roots || []).map((path) => ({ name: path, path })),
      files: [],
    });
  } catch (error) {
    setStatus(`Failed to load roots: ${error.message}`, true);
  }
}

async function loadPickerPath(path, mode) {
  try {
    const data = await apiJson(`/api/fs/list?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}`);
    renderPickerList(data);
  } catch (error) {
    setStatus(`Failed to list path: ${error.message}`, true);
  }
}

function closePicker() {
  state.picker.target = null;
  state.picker.currentPath = null;
  state.picker.parentPath = null;
  setPickerVisible(false);
}

function openPicker(target) {
  const config = PICKER_CONFIG[target];
  if (!config) {
    return;
  }
  state.picker.target = target;
  el.pickerTitle.textContent = config.title;
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
updatePointButtons();
setMode("image");
setStatus("Set a dataset folder or a video path, then click Load.");
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
    subtitle.textContent = `Area ${Math.round(polygonArea(segment.points))} px`;

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
      throw new Error(`Invalid response: ${text}`);
    }
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Request failed (${response.status})`;
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
    setStatus("No images found in the dataset.", true);
  }
}

async function loadData() {
  const videoPath = el.videoPathInput.value.trim();
  const datasetDir = el.datasetDirInput.value.trim();
  if (videoPath && datasetDir) {
    setStatus("Both dataset and video are set. Loading video.");
    return loadVideo();
  }
  if (videoPath) {
    return loadVideo();
  }
  if (datasetDir) {
    return loadSession();
  }
  setStatus("Dataset folder or video path is required.", true);
}

async function loadSession() {
  const datasetDir = el.datasetDirInput.value.trim();
  if (!datasetDir) {
    setStatus("Dataset path is required.", true);
    return;
  }
  const payload = {
    dataset_dir: datasetDir,
    classes: parseClassesFromInput(),
    classes_file: el.classesFileInput.value.trim() || null,
    model_path: el.modelPathInput.value.trim() || null,
    labels_dir: el.labelsDirInput.value.trim() || null,
  };
  setStatus("Loading dataset...");
  try {
    const data = await apiJson("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await applySessionData(data, "image");
  } catch (error) {
    setStatus(`Failed to load dataset: ${error.message}`, true);
  }
}

async function loadVideo() {
  const videoPath = el.videoPathInput.value.trim();
  if (!videoPath) {
    setStatus("Video path is required.", true);
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
  setStatus("Extracting frames from video...");
  try {
    const data = await apiJson("/api/video/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await applySessionData(data, "video");
    setStatus("Video loaded. Use frames like images to annotate.");
  } catch (error) {
    setStatus(`Failed to load video: ${error.message}`, true);
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
    updateStatusForCurrent("cached");
    render();
  } else {
    state.segments = [];
    refreshSegmentList();
  }

  const token = ++state.loadToken;
  setStatus("Loading image...");
  try {
    const img = new Image();
    const imageUrl = `/api/image?path=${encodeURIComponent(state.imagePath)}`;
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Failed to load image."));
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
          updateStatusForCurrent("copied from previous frame");
          return;
        }
      }
      updateStatusForCurrent("loaded");
    } catch (error) {
      setStatus(`Failed to load image: ${error.message}`, true);
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
  setStatus("No unlabeled images found.");
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
    setStatus("No image selected.", true);
    return;
  }
  const payload = buildPredictPayload();
  setStatus("Running auto segmentation...");
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
    setStatus(`Segmentation complete: masks ${state.segments.length}. Review the list.`);
  } catch (error) {
    setStatus(`Segmentation failed: ${error.message}`, true);
  }
}

async function segmentFromPoints() {
  if (!state.imagePath) {
    setStatus("No image selected.", true);
    return;
  }
  if (!state.points.items.length) {
    setStatus("Add some points first.", true);
    return;
  }
  if (!state.points.items.some((point) => point.label === 1)) {
    setStatus("Add at least one positive point.", true);
    return;
  }
  if ((el.engineSelect.value || "sam") !== "sam2") {
    setStatus("SAM 2 is required. Select it in the segmentation settings.", true);
    return;
  }
  if (!el.sam2ConfigInput.value.trim()) {
    setStatus("SAM 2 YAML config is required.", true);
    return;
  }
  if (!el.modelPathInput.value.trim()) {
    setStatus("SAM 2 model path is required.", true);
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

  setStatus("Segmenting from points...");
  try {
    const data = await apiJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const predicted = normalizeSegments(data.segments || []);
    if (!predicted.length) {
      setStatus("No masks returned.", true);
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
        setStatus("Failed to choose a primary mask to refine.", true);
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
      setStatus("Mask updated from points. Add more points to refine.");
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
      setStatus(`Added masks from points: ${predicted.length}.`);
    }
  } catch (error) {
    setStatus(`Point segmentation failed: ${error.message}`, true);
  }
}

async function propagateVideo() {
  if (state.mode !== "video") {
    setStatus("Load a video first.", true);
    return;
  }
  if ((el.engineSelect.value || "sam") !== "sam2") {
    setStatus("SAM 2 is required.", true);
    return;
  }
  if (!el.sam2ConfigInput.value.trim()) {
    setStatus("SAM 2 YAML config is required for propagation.", true);
    return;
  }
  if (!el.modelPathInput.value.trim()) {
    setStatus("SAM 2 model path is required for propagation.", true);
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
  setStatus("Propagating mask through video... this may take a while.");
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
    setStatus(`Propagation complete: saved ${data.saved_count}, errors ${data.error_count}.`);
  } catch (error) {
    setStatus(`Propagation failed: ${error.message}`, true);
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
    setStatus("No image selected.", true);
    return;
  }
  const keptSegments = state.segments.filter((segment) => segment.keep !== false);
  const payload = {
    segments: toPayloadSegments(state.segments),
  };
  setStatus("Saving...");
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
    updateStatusForCurrent(`Saved to ${result.label_path}`);
  } catch (error) {
    setStatus(`Save failed: ${error.message}`, true);
  }
}

async function saveAllDirty() {
  const dirtyPaths = Array.from(state.dirtyPaths);
  if (dirtyPaths.length === 0) {
    setStatus("Nothing to save.");
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
  setStatus(`Batch saving: ${items.length}...`);
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
    updateStatusForCurrent(`Saved: ${result.saved_count}, errors ${result.error_count}`);
  } catch (error) {
    setStatus(`Batch save failed: ${error.message}`, true);
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
    setStatus(`Removed unchecked masks: ${before - state.segments.length}.`);
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
    setStatus("No mask selected to delete.");
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
  const dirty = state.dirty ? " (unsaved)" : "";
  const kept = countKeptSegments(state.segments);
  const tail = extra ? ` | ${extra}` : "";
  setStatus(
    `${state.imagePath} | ${state.imageIndex + 1}/${state.images.length} | masks: ${state.segments.length} (kept ${kept})${dirty} | unsaved: ${state.dirtyPaths.size}${tail}`
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
