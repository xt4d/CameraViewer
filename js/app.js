import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { unzipSync } from "fflate";

const datasetInput = document.getElementById("dataset");
const datasetHint = document.getElementById("dataset-hint");
const chooseFolderButton = document.getElementById("choose-folder");
const formatSelect = document.getElementById("format");
const poseSelect = document.getElementById("pose-type");
const poseField = document.querySelector('[data-field="pose"]');
const imageSizeInput = document.getElementById("image-size");
const sceneSizeInput = document.getElementById("scene-size");
const rescaleInput = document.getElementById("rescale");
const showImagesInput = document.getElementById("show-images");
const upAxisSelect = document.getElementById("up-axis");
const showAxesInput = document.getElementById("show-axes");
const showGridInput = document.getElementById("show-grid");
const recenterInput = document.getElementById("recenter");
const coordSystemInputs = document.querySelectorAll('input[name="coord-system"]');
const renderButton = document.getElementById("render");
const resetButton = document.getElementById("reset");
const statusEl = document.getElementById("status");
const errorsEl = document.getElementById("errors");
const plotEl = document.getElementById("plot");

const state = {
  entries: [],
  fileMap: new Map(),
  root: "",
  hasRendered: false,
  autoParams: null,
  formatAutoDetected: false,
  coordSystemTouched: false,
};

const BASE_IMAGE_SIZE = 128;
const QUANT_LEVELS = { r: 16, g: 16, b: 16 };
const ENABLE_QUANTIZATION = true;

const viewer = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  content: null,
  resizeObserver: null,
  upAxis: "z-up",
  helpers: null,
};


function setStatus(msg) {
  statusEl.textContent = msg || "";
}


function setErrors(errors) {
  if (!errors || errors.length === 0) {
    errorsEl.hidden = true;
    errorsEl.innerHTML = "";
    return;
  }
  errorsEl.hidden = false;
  errorsEl.innerHTML = `<strong>Heads up:</strong><ul>${errors
    .map((err) => `<li>${err}</li>`)
    .join("")}</ul>`;
}


function normalizePath(path) {
  return path.replace(/\\/g, "/");
}


function detectRootsFromEntries(entries) {
  const roots = new Set();
  entries.forEach(({ path }) => {
    if (path.includes("/")) {
      roots.add(path.split("/")[0]);
    } else {
      roots.add("");
    }
  });
  return roots;
}


function normalizeEntriesFromInput(fileList) {
  const files = Array.from(fileList || []);
  const entries = files.map((file) => ({
    file,
    path: normalizePath(file.webkitRelativePath || file.name),
  }));

  const roots = detectRootsFromEntries(entries);
  const root = roots.values().next().value || "";
  const trimmed = entries.map(({ file, path }) => ({
    file,
    path: root && path.startsWith(`${root}/`) ? path.slice(root.length + 1) : path,
  }));

  return { entries: trimmed, root, roots };
}


function buildFileMap(entries) {
  const map = new Map();
  entries.forEach(({ file, path }) => {
    map.set(path, file);
  });
  return map;
}


async function collectFilesFromHandle(dirHandle, prefix = "") {
  const entries = [];
  for await (const [name, handle] of dirHandle.entries()) {
    if (handle.kind === "file") {
      const file = await handle.getFile();
      entries.push({ file, path: normalizePath(`${prefix}${name}`) });
    } else if (handle.kind === "directory") {
      const sub = await collectFilesFromHandle(handle, `${prefix}${name}/`);
      entries.push(...sub);
    }
  }
  return entries;
}


function setDatasetEntries(entries, rootLabel, roots = null) {
  state.entries = entries;
  state.fileMap = buildFileMap(entries);
  state.root = rootLabel || "";
  state.hasRendered = false;
  state.autoParams = null;
  state.formatAutoDetected = false;
  state.coordSystemTouched = false;
  const detected = detectFormatFromEntries();
  if (detected) {
    formatSelect.value = detected;
    updatePoseState();
    state.formatAutoDetected = true;
  }
  if (roots && roots.size > 1) {
    setErrors(["Please choose a single scene folder."]);
  } else {
    setErrors([]);
  }
  datasetHint.textContent = state.root
    ? `Selected: ${state.root} (Files stay local).`
    : `Selected (Files stay local).`;
}


function revealLegacyPicker(note) {
  datasetInput.classList.remove("is-hidden");
  if (note) {
    setErrors([note]);
  }
  datasetHint.textContent = "Use the legacy picker (Files stay local).";
}


function updatePoseState() {
  const disabled = formatSelect.value !== "quick";
  poseSelect.disabled = disabled;
  poseField.classList.toggle("is-disabled", disabled);
  setCoordSystemDefault(formatSelect.value);
}


function initViewer(upAxis = "z-up") {
  if (viewer.renderer) {
    ensureControlsForUpAxis(upAxis);
    return viewer;
  }

  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

  plotEl.innerHTML = "";
  const width = plotEl.clientWidth || 900;
  const height = plotEl.clientHeight || 560;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  plotEl.appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 2000);
  camera.position.set(1.5, 1.5, 1.0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.85);
  const directional = new THREE.DirectionalLight(0xffffff, 0.35);
  directional.position.set(2, 3, 4);
  scene.add(ambient, directional);

  const content = new THREE.Group();
  scene.add(content);

  renderer.setAnimationLoop(() => {
    if (viewer.controls) {
      viewer.controls.update();
    }
    renderer.render(scene, camera);
  });

  const resizeObserver = new ResizeObserver(() => {
    if (!viewer.renderer) {
      return;
    }
    const w = plotEl.clientWidth || 900;
    const h = plotEl.clientHeight || 560;
    viewer.camera.aspect = w / h;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(w, h);
  });
  resizeObserver.observe(plotEl);

  viewer.renderer = renderer;
  viewer.scene = scene;
  viewer.camera = camera;
  viewer.content = content;
  viewer.resizeObserver = resizeObserver;

  ensureControlsForUpAxis(upAxis);

  return viewer;
}


function clearContent() {
  if (!viewer.content) {
    return;
  }
  viewer.helpers = null;
  while (viewer.content.children.length > 0) {
    const child = viewer.content.children.pop();
    viewer.content.remove(child);
    disposeObject(child);
  }
}


function disposeObject(object) {
  if (!object) {
    return;
  }
  if (object.geometry) {
    object.geometry.dispose();
  }
  if (object.material) {
    if (Array.isArray(object.material)) {
      object.material.forEach((mat) => disposeMaterial(mat));
    } else {
      disposeMaterial(object.material);
    }
  }
  if (object.children && object.children.length > 0) {
    object.children.forEach((child) => disposeObject(child));
  }
}


function disposeMaterial(material) {
  if (material.map) {
    material.map.dispose();
  }
  material.dispose();
}


function showPlaceholder() {
  plotEl.innerHTML = `<div class="placeholder">
    <p>Render a scene to see the interactive plot here.</p>
    <span>Use the mouse to orbit, pan, and zoom.</span>
  </div>`;
}


function teardownViewer() {
  if (viewer.resizeObserver) {
    viewer.resizeObserver.disconnect();
  }
  if (viewer.controls) {
    viewer.controls.dispose();
  }
  if (viewer.renderer) {
    viewer.renderer.setAnimationLoop(null);
    viewer.renderer.dispose();
  }
  viewer.renderer = null;
  viewer.scene = null;
  viewer.camera = null;
  viewer.controls = null;
  viewer.content = null;
  viewer.resizeObserver = null;
  viewer.upAxis = "z-up";
  viewer.helpers = null;
  showPlaceholder();
}


function buildHelpers(sceneSize, upAxis, showAxes = true, showGrid = true) {
  const group = new THREE.Group();
  const grid = new THREE.GridHelper(sceneSize * 2, sceneSize * 2, 0xb7b1a6, 0xd9d2c6);
  orientGridForUpAxis(grid, upAxis);
  grid.position.z = 0;
  const axesLength = sceneSize;
  const axes = new THREE.AxesHelper(axesLength);
  const axesGroup = new THREE.Group();
  axesGroup.add(axes, buildAxisAnnotations(axesLength, sceneSize));
  grid.visible = showGrid;
  axesGroup.visible = showAxes;
  group.add(grid, axesGroup);
  group.userData.helpers = { grid, axesGroup };
  return group;
}


function updateHelperVisibility() {
  if (!viewer.helpers) {
    return;
  }
  const showAxes = showAxesInput ? showAxesInput.checked : true;
  const showGrid = showGridInput ? showGridInput.checked : true;
  if (viewer.helpers.axesGroup) {
    viewer.helpers.axesGroup.visible = showAxes;
  }
  if (viewer.helpers.grid) {
    viewer.helpers.grid.visible = showGrid;
  }
}


function orientGridForUpAxis(grid, upAxis) {
  switch (upAxis) {
    case "x-up":
      grid.rotation.z = -Math.PI / 2;
      break;
    case "x-down":
      grid.rotation.z = Math.PI / 2;
      break;
    case "y-down":
      grid.rotation.x = Math.PI;
      break;
    case "y-up":
      break;
    case "z-down":
      grid.rotation.x = -Math.PI / 2;
      break;
    case "z-up":
    default:
      grid.rotation.x = Math.PI / 2;
      break;
  }
}


function getUpAxisVector(upAxis) {
  switch (upAxis) {
    case "x-up":
      return [1, 0, 0];
    case "x-down":
      return [-1, 0, 0];
    case "y-up":
      return [0, 1, 0];
    case "y-down":
      return [0, -1, 0];
    case "z-down":
      return [0, 0, -1];
    case "z-up":
    default:
      return [0, 0, 1];
  }
}


function getCameraPositionForUpAxis(upAxis, distance, height) {
  switch (upAxis) {
    case "x-up":
      return [height, distance, distance];
    case "x-down":
      return [-height, distance, distance];
    case "y-up":
      return [distance, height, distance];
    case "y-down":
      return [distance, -height, distance];
    case "z-down":
      return [distance, distance, -height];
    case "z-up":
    default:
      return [distance, distance, height];
  }
}


function updateCameraForScene(sceneSize, upAxis = "z-up") {
  if (!viewer.camera || !viewer.renderer) {
    return;
  }
  ensureControlsForUpAxis(upAxis);
  const distance = Math.max(sceneSize * 1.8, 1.5);
  const height = sceneSize * 1.2;
  viewer.camera.near = Math.max(sceneSize * 0.001, 0.01);
  viewer.camera.far = sceneSize * 20 + 100;
  const upVector = getUpAxisVector(upAxis);
  const position = getCameraPositionForUpAxis(upAxis, distance, height);
  viewer.camera.position.set(position[0], position[1], position[2]);
  viewer.camera.up.set(upVector[0], upVector[1], upVector[2]);
  viewer.controls.target.set(0, 0, 0);
  viewer.controls.update();
}


function ensureControlsForUpAxis(upAxis) {
  if (!viewer.camera || !viewer.renderer) {
    return;
  }
  if (!OrbitControls) {
    throw new Error("OrbitControls did not load. Check the script tag.");
  }
  const upVector = getUpAxisVector(upAxis);
  const needsRebuild = !viewer.controls || viewer.upAxis !== upAxis;
  if (needsRebuild) {
    if (viewer.controls) {
      viewer.controls.dispose();
    }
    viewer.camera.up.set(upVector[0], upVector[1], upVector[2]);
    const controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    viewer.controls = controls;
    viewer.upAxis = upAxis;
  } else {
    viewer.camera.up.set(upVector[0], upVector[1], upVector[2]);
  }
}


function poseToMatrix4(pose) {
  const mat = new THREE.Matrix4();
  mat.set(
    pose[0][0],
    pose[0][1],
    pose[0][2],
    pose[0][3],
    pose[1][0],
    pose[1][1],
    pose[1][2],
    pose[1][3],
    pose[2][0],
    pose[2][1],
    pose[2][2],
    pose[2][3],
    pose[3][0],
    pose[3][1],
    pose[3][2],
    pose[3][3]
  );
  return mat;
}


function createConeLines(points, color) {
  const edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 1],
    [0, 5],
  ];
  const vertices = [];
  edges.forEach((edge) => {
    const a = points[edge[0]];
    const b = points[edge[1]];
    vertices.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const material = new THREE.LineBasicMaterial({ color });
  return new THREE.LineSegments(geometry, material);
}


function createTextSprite(text, color, scaleFactor = 1) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const fontSize = 32;
  ctx.font = `${fontSize}px "Space Grotesk", sans-serif`;
  const metrics = ctx.measureText(text);
  const padding = 10;
  canvas.width = metrics.width + padding * 2;
  canvas.height = fontSize + padding * 2;
  ctx.font = `${fontSize}px "Space Grotesk", sans-serif`;
  ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = color;
  ctx.fillText(text, padding, fontSize + padding / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  const scale = 0.02 * canvas.width * scaleFactor;
  sprite.scale.set(scale, scale * (canvas.height / canvas.width), 1);
  return sprite;
}


function buildAxisAnnotations(axisLength, tickCount = 5) {
  const group = new THREE.Group();
  const tickSize = axisLength * 0.03;
  const vertices = [];
  const step = axisLength / tickCount;

  for (let idx = 1; idx <= tickCount; idx += 1) {
    const pos = step * idx;
    vertices.push(pos, -tickSize, 0, pos, tickSize, 0);
    vertices.push(-tickSize, pos, 0, tickSize, pos, 0);
    vertices.push(-tickSize, 0, pos, tickSize, 0, pos);
  }

  if (vertices.length > 0) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    const material = new THREE.LineBasicMaterial({
      color: 0x8f877b,
      transparent: true,
      opacity: 0.85,
    });
    group.add(new THREE.LineSegments(geometry, material));
  }

  const labelOffset = tickSize * 2.4;
  const labelScale = 0.55;
  const xLabel = createTextSprite("X", "#dc2626", labelScale);
  xLabel.position.set(axisLength + labelOffset, 0, 0);
  const yLabel = createTextSprite("Y", "#16a34a", labelScale);
  yLabel.position.set(0, axisLength + labelOffset, 0);
  const zLabel = createTextSprite("Z", "#2563eb", labelScale);
  zLabel.position.set(0, 0, axisLength + labelOffset);
  group.add(xLabel, yLabel, zLabel);

  return group;
}


async function readTextFile(path) {
  const file = state.fileMap.get(path);
  if (!file) {
    throw new Error(`Missing file: ${path}`);
  }
  return file.text();
}


async function readJsonFile(path) {
  const text = await readTextFile(path);
  return JSON.parse(text);
}


async function readBinaryFile(path) {
  const file = state.fileMap.get(path);
  if (!file) {
    throw new Error(`Missing file: ${path}`);
  }
  return file.arrayBuffer();
}


function listFilesInDir(prefix) {
  const files = [];
  for (const key of state.fileMap.keys()) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    const rest = key.slice(prefix.length);
    if (rest && !rest.includes("/")) {
      files.push(key);
    }
  }
  return files.sort();
}


function listFilesByExtension(extensions) {
  const lowered = extensions.map((ext) => ext.toLowerCase());
  return Array.from(state.fileMap.keys()).filter((key) => {
    const lower = key.toLowerCase();
    return lowered.some((ext) => lower.endsWith(ext));
  });
}


function detectFormatFromEntries() {
  const keys = Array.from(state.fileMap.keys());
  const hasQuickPoses =
    state.fileMap.has("poses.json") ||
    keys.some((key) => key.endsWith("/poses.json") || key.startsWith("poses/"));
  if (hasQuickPoses) {
    return "quick";
  }
  const hasNerfTransforms = keys.some(
    (key) =>
      key.endsWith("transforms.json") ||
      key.endsWith("transforms_train.json") ||
      key.endsWith("transforms_test.json")
  );
  if (hasNerfTransforms) {
    return "nerf";
  }
  const hasColmapImages = keys.some(
    (key) => key === "images.txt" || key.endsWith("/images.txt")
  );
  if (hasColmapImages) {
    return "colmap";
  }
  const hasNpy = keys.some(
    (key) => key.toLowerCase().endsWith(".npy") || key.toLowerCase().endsWith(".npz")
  );
  if (hasNpy) {
    return "npy";
  }
  return null;
}


function parseTextMatrix(text) {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error("Pose file is empty.");
  }
  const rows = lines.map((line) =>
    line
      .split(/[\s,]+/)
      .filter(Boolean)
      .map((val) => Number(val))
  );
  if (rows.length === 1) {
    return rows[0];
  }
  return rows;
}


function parseNpy(buffer, options = {}) {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  if (magic[0] !== 0x93 || String.fromCharCode(...magic.slice(1)) !== "NUMPY") {
    throw new Error("Invalid NPY file.");
  }
  const major = view.getUint8(6);
  const minor = view.getUint8(7);
  let headerLen;
  let offset;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10;
  } else {
    headerLen = view.getUint32(8, true);
    offset = 12;
  }
  const headerText = new TextDecoder("ascii").decode(
    new Uint8Array(buffer, offset, headerLen)
  );
  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = headerText.match(/'fortran_order':\s*(False|True)/);
  const shapeMatch = headerText.match(/'shape':\s*\(([^\)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error("Unsupported NPY header.");
  }
  const dtype = descrMatch[1];
  const fortran = fortranMatch[1] === "True";
  const shape = shapeMatch[1]
    .split(",")
    .map((val) => val.trim())
    .filter(Boolean)
    .map((val) => Number(val));

  const typeChar = dtype[dtype.length - 1];
  const byteOrder = dtype[0];
  const typeBytes = Number(dtype.slice(2));

  if (byteOrder !== "<" && byteOrder !== "|") {
    throw new Error("Only little-endian NPY files are supported.");
  }
  // if (typeChar !== "f") {
  //   throw new Error("Only float NPY files are supported.");
  // }

  const dataOffset = offset + headerLen;
  const total = shape.reduce((acc, dim) => acc * dim, 1);
  const data = new Array(total);
  for (let i = 0; i < total; i += 1) {
    const byteIndex = dataOffset + i * typeBytes;
    if (typeBytes === 4) {
      data[i] = view.getFloat32(byteIndex, true);
    } else if (typeBytes === 8) {
      data[i] = view.getFloat64(byteIndex, true);
    } else {
      throw new Error("Unsupported float size in NPY file.");
    }
  }

  if (fortran && shape.length > 1) {
    const reshaped = reshapeFortran(data, shape);
    return options.returnShape ? { data: reshaped, shape } : reshaped;
  }
  const reshaped = reshapeRowMajor(data, shape);
  return options.returnShape ? { data: reshaped, shape } : reshaped;
}


function reshapeRowMajor(data, shape) {
  return reshapeND(data, shape);
}


function reshapeFortran(data, shape) {
  if (shape.length === 1) {
    return data.slice();
  }
  const total = shape.reduce((acc, dim) => acc * dim, 1);
  const reordered = new Array(total);
  for (let idx = 0; idx < total; idx += 1) {
    let remainder = idx;
    let rowMajorIndex = 0;
    for (let dim = 0; dim < shape.length; dim += 1) {
      const size = shape[dim];
      const coord = remainder % size;
      remainder = Math.floor(remainder / size);
      rowMajorIndex = rowMajorIndex * size + coord;
    }
    reordered[rowMajorIndex] = data[idx];
  }
  return reshapeND(reordered, shape);
}


function reshapeND(data, shape) {
  if (shape.length === 1) {
    return data.slice(0, shape[0]);
  }
  let offset = 0;
  const build = (dim) => {
    const size = shape[dim];
    if (dim === shape.length - 1) {
      const slice = data.slice(offset, offset + size);
      offset += size;
      return slice;
    }
    const out = [];
    for (let i = 0; i < size; i += 1) {
      out.push(build(dim + 1));
    }
    return out;
  };
  return build(0);
}


function asMatrix(mat, rows, cols) {
  if (Array.isArray(mat) && Array.isArray(mat[0])) {
    return mat.map((row) => row.map((val) => Number(val)));
  }
  const flat = Array.isArray(mat) ? mat.map((val) => Number(val)) : [Number(mat)];
  if (rows && cols) {
    const out = [];
    for (let r = 0; r < rows; r += 1) {
      out.push(flat.slice(r * cols, r * cols + cols));
    }
    return out;
  }
  return [flat];
}


function ensureMatrix4(mat) {
  if (Array.isArray(mat) && Array.isArray(mat[0])) {
    const rows = mat.length;
    const cols = mat[0].length;
    if (rows === 4 && cols === 4) {
      return mat.map((row) => row.slice());
    }
    if (rows === 3 && cols === 4) {
      return [...mat.map((row) => row.slice()), [0, 0, 0, 1]];
    }
  }
  const flat = Array.isArray(mat) ? mat.flat() : [Number(mat)];
  if (flat.length === 16) {
    return asMatrix(flat, 4, 4);
  }
  if (flat.length === 12) {
    return [...asMatrix(flat, 3, 4), [0, 0, 0, 1]];
  }
  return null;
}

const OPENCV_TO_OPENGL = [
  [1, 0, 0, 0],
  [0, -1, 0, 0],
  [0, 0, -1, 0],
  [0, 0, 0, 1],
];

function convertPoseFromOpenCV(c2w) {
  return mat4Mul(OPENCV_TO_OPENGL, mat4Mul(c2w, OPENCV_TO_OPENGL));
}


function alignPosesUp(poses) {
  if (!poses || poses.length === 0) {
    return poses;
  }
  let up = [0, 0, 0];
  poses.forEach((pose) => {
    up = add(up, [pose[0][1], pose[1][1], pose[2][1]]);
  });
  up = normalize(up);
  const upRot = rotmat(up, [0, 0, 1]);
  const upRot4 = [
    [upRot[0][0], upRot[0][1], upRot[0][2], 0],
    [upRot[1][0], upRot[1][1], upRot[1][2], 0],
    [upRot[2][0], upRot[2][1], upRot[2][2], 0],
    [0, 0, 0, 1],
  ];
  return poses.map((pose) => mat4Mul(upRot4, pose));
}


function selectNpyCandidate(candidates) {
  if (!candidates || candidates.length === 0) {
    throw new Error("No .npy or .npz file found for NPY format.");
  }
  const sorted = candidates.slice().sort();
  const preferred = sorted.find((name) => /poses\.(npy|npz)$/i.test(name));
  if (preferred) {
    return preferred;
  }
  if (sorted.length === 1) {
    return sorted[0];
  }
  throw new Error(
    `Multiple numpy files found (${sorted.slice(0, 3).join(", ")}). Keep one or rename to poses.npy.`
  );
}


function extractNpyFromNpz(buffer) {
  const entries = unzipSync(new Uint8Array(buffer));
  const names = Object.keys(entries).filter((name) => name.toLowerCase().endsWith(".npy"));
  if (names.length === 0) {
    throw new Error("NPZ file does not contain any .npy arrays.");
  }
  const preferred = names.find((name) => /poses\.npy$/i.test(name));
  const target = preferred || names[0];
  const bytes = entries[target];
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}


function posesFromNpyTensor(tensor, shape) {
  if (!shape || shape.length === 0) {
    throw new Error("NPY tensor has no shape.");
  }
  if (shape.length === 2) {
    const pose = ensureMatrix4(tensor);
    if (!pose) {
      throw new Error("NPY tensor must be 3x4 or 4x4.");
    }
    return [pose];
  }
  if (shape.length !== 3) {
    throw new Error("NPY tensor must have shape [num, 3, 4] or [num, 4, 4].");
  }
  const [, rows, cols] = shape;
  if (!((rows === 3 && cols === 4) || (rows === 4 && cols === 4))) {
    throw new Error("NPY tensor must have shape [num, 3, 4] or [num, 4, 4].");
  }
  return tensor
    .map((mat) => ensureMatrix4(mat))
    .filter((mat) => mat !== null);
}


function ensureVector3(mat) {
  if (Array.isArray(mat) && !Array.isArray(mat[0])) {
    if (mat.length >= 3) {
      return [Number(mat[0]), Number(mat[1]), Number(mat[2])];
    }
  }
  const flat = Array.isArray(mat) ? mat.flat() : [Number(mat)];
  if (flat.length >= 3) {
    return [flat[0], flat[1], flat[2]];
  }
  throw new Error("Expected a 3D vector.");
}


function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}


function norm(a) {
  return Math.sqrt(dot(a, a));
}


function normalize(a) {
  const n = norm(a);
  if (n < 1e-8) {
    return [a[0], a[1], a[2]];
  }
  return [a[0] / n, a[1] / n, a[2] / n];
}


function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}


function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}


function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}


function mat3MulVec3(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}


function mat4Mul(a, b) {
  const out = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      out[r][c] =
        a[r][0] * b[0][c] +
        a[r][1] * b[1][c] +
        a[r][2] * b[2][c] +
        a[r][3] * b[3][c];
    }
  }
  return out;
}


function mat4Invert(m) {
  const a00 = m[0][0];
  const a01 = m[0][1];
  const a02 = m[0][2];
  const a03 = m[0][3];
  const a10 = m[1][0];
  const a11 = m[1][1];
  const a12 = m[1][2];
  const a13 = m[1][3];
  const a20 = m[2][0];
  const a21 = m[2][1];
  const a22 = m[2][2];
  const a23 = m[2][3];
  const a30 = m[3][0];
  const a31 = m[3][1];
  const a32 = m[3][2];
  const a33 = m[3][3];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det =
    b00 * b11 -
    b01 * b10 +
    b02 * b09 +
    b03 * b08 -
    b04 * b07 +
    b05 * b06;

  if (!det) {
    throw new Error("Matrix is not invertible.");
  }
  det = 1.0 / det;

  const out = [
    [
      (a11 * b11 - a12 * b10 + a13 * b09) * det,
      (a02 * b10 - a01 * b11 - a03 * b09) * det,
      (a31 * b05 - a32 * b04 + a33 * b03) * det,
      (a22 * b04 - a21 * b05 - a23 * b03) * det,
    ],
    [
      (a12 * b08 - a10 * b11 - a13 * b07) * det,
      (a00 * b11 - a02 * b08 + a03 * b07) * det,
      (a32 * b02 - a30 * b05 - a33 * b01) * det,
      (a20 * b05 - a22 * b02 + a23 * b01) * det,
    ],
    [
      (a10 * b10 - a11 * b08 + a13 * b06) * det,
      (a01 * b08 - a00 * b10 - a03 * b06) * det,
      (a30 * b04 - a31 * b02 + a33 * b00) * det,
      (a21 * b02 - a20 * b04 - a23 * b00) * det,
    ],
    [
      (a11 * b07 - a10 * b09 - a12 * b06) * det,
      (a00 * b09 - a01 * b07 + a02 * b06) * det,
      (a31 * b01 - a30 * b03 - a32 * b00) * det,
      (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ],
  ];

  return out;
}


function sphericalToCartesian(sph) {
  const [theta, azimuth, radius] = sph;
  return [
    radius * Math.sin(theta) * Math.cos(azimuth),
    radius * Math.sin(theta) * Math.sin(azimuth),
    radius * Math.cos(theta),
  ];
}


function eluToC2w(eye, lookat, up) {
  const eyeV = ensureVector3(eye);
  const lookatV = ensureVector3(lookat);
  const upV = ensureVector3(up);

  let l = sub(eyeV, lookatV);
  if (norm(l) < 1e-8) {
    l = [l[0], l[1], 1];
  }
  l = normalize(l);

  let s = cross(l, upV);
  if (norm(s) < 1e-8) {
    s = [1, 0, 0];
  }
  s = normalize(s);
  const uu = cross(s, l);

  const rot = [
    [-s[0], -s[1], -s[2]],
    [uu[0], uu[1], uu[2]],
    [l[0], l[1], l[2]],
  ];

  const c2w = [
    [rot[0][0], rot[1][0], rot[2][0], eyeV[0]],
    [rot[0][1], rot[1][1], rot[2][1], eyeV[1]],
    [rot[0][2], rot[1][2], rot[2][2], eyeV[2]],
    [0, 0, 0, 1],
  ];

  return c2w;
}


function qvecToRotmat(qvec) {
  return [
    [
      1 - 2 * qvec[2] * qvec[2] - 2 * qvec[3] * qvec[3],
      2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
      2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
    ],
    [
      2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
      1 - 2 * qvec[1] * qvec[1] - 2 * qvec[3] * qvec[3],
      2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
    ],
    [
      2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
      2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
      1 - 2 * qvec[1] * qvec[1] - 2 * qvec[2] * qvec[2],
    ],
  ];
}


function rotmat(a, b) {
  const aN = normalize(a);
  const bN = normalize(b);
  let v = cross(aN, bN);
  let c = dot(aN, bN);
  if (c < -1 + 1e-10) {
    return rotmat([aN[0] + (Math.random() - 0.5) * 1e-2, aN[1], aN[2]], bN);
  }
  const s = norm(v);
  const kmat = [
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0],
  ];

  const kmat2 = mat3Mul(kmat, kmat);
  const scaleFactor = (1 - c) / (s * s + 1e-10);
  return [
    [1 + kmat[0][0] + kmat2[0][0] * scaleFactor, kmat[0][1] + kmat2[0][1] * scaleFactor, kmat[0][2] + kmat2[0][2] * scaleFactor],
    [kmat[1][0] + kmat2[1][0] * scaleFactor, 1 + kmat[1][1] + kmat2[1][1] * scaleFactor, kmat[1][2] + kmat2[1][2] * scaleFactor],
    [kmat[2][0] + kmat2[2][0] * scaleFactor, kmat[2][1] + kmat2[2][1] * scaleFactor, 1 + kmat[2][2] + kmat2[2][2] * scaleFactor],
  ];
}


function mat3Mul(a, b) {
  return [
    [
      a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
      a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
      a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
    ],
    [
      a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
      a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
      a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
    ],
    [
      a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
      a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
      a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
    ],
  ];
}


function recenterCameras(c2ws) {
  const centers = c2ws.map((c2w) => [c2w[0][3], c2w[1][3], c2w[2][3]]);
  const mean = centers.reduce(
    (acc, val) => [acc[0] + val[0], acc[1] + val[1], acc[2] + val[2]],
    [0, 0, 0]
  );
  const count = centers.length;
  const center = [mean[0] / count, mean[1] / count, mean[2] / count];
  return c2ws.map((c2w) => {
    const out = c2w.map((row) => row.slice());
    out[0][3] -= center[0];
    out[1][3] -= center[1];
    out[2][3] -= center[2];
    return out;
  });
}


function rescaleCameras(c2ws, scaleValue) {
  return c2ws.map((c2w) => {
    const out = c2w.map((row) => row.slice());
    out[0][3] *= scaleValue;
    out[1][3] *= scaleValue;
    out[2][3] *= scaleValue;
    return out;
  });
}


function getPoseBounds(c2ws) {
  if (!c2ws || c2ws.length === 0) {
    return null;
  }
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  c2ws.forEach((c2w) => {
    const x = c2w[0][3];
    const y = c2w[1][3];
    const z = c2w[2][3];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  });
  const sizeX = maxX - minX;
  const sizeY = maxY - minY;
  const sizeZ = maxZ - minZ;
  const radius = 0.5 * Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ);
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    size: [sizeX, sizeY, sizeZ],
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
    radius,
  };
}


function computeAutoScaleAndScene(c2ws) {
  const bounds = getPoseBounds(c2ws);
  if (!bounds || !Number.isFinite(bounds.radius) || bounds.radius < 1e-6) {
    return { scale: 1.0, sceneSize: 5.0, bounds, scaledRadius: 0 };
  }
  const sceneSize = Math.max(bounds.radius * 1.25, 2.0);
  let scale = 1.0;
  return { scale, sceneSize, bounds };
}


function calcCamConePts3d(c2w, fovDeg, zoom = 1.0) {
  const fovRad = (fovDeg * Math.PI) / 180.0;

  const camX = c2w[0][3];
  const camY = c2w[1][3];
  const camZ = c2w[2][3];

  const tanVal = Math.tan(fovRad / 2.0);
  const corn1 = [tanVal, tanVal, -1.0];
  const corn2 = [-tanVal, tanVal, -1.0];
  const corn3 = [-tanVal, -tanVal, -1.0];
  const corn4 = [tanVal, -tanVal, -1.0];
  const corn5 = [0, tanVal, -1.0];

  const rot = [
    [c2w[0][0], c2w[0][1], c2w[0][2]],
    [c2w[1][0], c2w[1][1], c2w[1][2]],
    [c2w[2][0], c2w[2][1], c2w[2][2]],
  ];

  const corners = [corn1, corn2, corn3, corn4, corn5].map((corn) => {
    const rotated = mat3MulVec3(rot, corn);
    const scaled = scale(rotated, zoom / norm(rotated));
    return [camX + scaled[0], camY + scaled[1], camZ + scaled[2]];
  });

  const points = [[camX, camY, camZ], ...corners];
  return points;
}


async function loadImageTexture(file, size) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, size, size);
  URL.revokeObjectURL(url);
  if (ENABLE_QUANTIZATION) {
    const imageData = ctx.getImageData(0, 0, size, size);
    quantizeImageData(imageData.data, QUANT_LEVELS);
    ctx.putImageData(imageData, 0, 0);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  texture.flipY = false;
  if (viewer.renderer) {
    texture.anisotropy = viewer.renderer.capabilities.getMaxAnisotropy();
  }
  return { texture, aspect: canvas.height / canvas.width };
}


function createImagePlane(pose, texture, aspect, imageScale = 1) {
  const width =  imageScale;
  const height =  aspect * imageScale;
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.position.set(0, 0, 1);
  plane.scale.y = -1;

  const holder = new THREE.Object3D();
  holder.add(plane);
  holder.matrixAutoUpdate = false;
  holder.matrix.copy(poseToMatrix4(pose));
  return holder;
}


function quantizeImageData(data, levels) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = quantizeChannel(data[i], levels.r);
    data[i + 1] = quantizeChannel(data[i + 1], levels.g);
    data[i + 2] = quantizeChannel(data[i + 2], levels.b);
  }
}


function quantizeChannel(value, levels) {
  if (levels <= 1) {
    return 0;
  }
  const idx = clampIndex(Math.round((value / 255) * (levels - 1)), 0, levels - 1);
  return Math.round((idx / (levels - 1)) * 255);
}


function clampIndex(value, min, max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}


function clampValue(value, min, max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}


function parseNumberOr(value, fallback) {
  if (value === "" || value === null || value === undefined) {
    return fallback;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}


function resolveImagePath(basePath) {
  if (!basePath) {
    return null;
  }
  const cleaned = basePath.replace(/^\.\//, "");
  if (state.fileMap.has(cleaned)) {
    return cleaned;
  }
  const exts = ["png", "jpg", "jpeg"];
  for (const ext of exts) {
    const candidate = `${cleaned}.${ext}`;
    if (state.fileMap.has(candidate)) {
      return candidate;
    }
  }
  return null;
}


async function loadQuick(poseType) {
  const poses = [];
  const legends = [];
  const colors = [];
  const imageFiles = [];

  let type = poseType === "auto" ? null : poseType;
  let frames;

  if (!type) {
    const json = await readJsonFile("poses.json");
    type = json.type;
    frames = json.frames || [];
  } else {
    frames = listFilesInDir("poses/");
  }

  for (let idx = 0; idx < frames.length; idx += 1) {
    const frame = frames[idx];
    let mat;
    let imagePath = null;
    let fid = idx;

    if (typeof frame === "string") {
      const fname = frame.split("/").pop();
      const parts = fname.split(".");
      fid = parts[0];
      const ext = parts[parts.length - 1];
      if (ext === "npy") {
        mat = parseNpy(await readBinaryFile(frame));
      } else {
        mat = parseTextMatrix(await readTextFile(frame));
      }
      const imageRoot = "images/";
      const imageCandidate = resolveImagePath(`${imageRoot}${fid}`);
      imagePath = imageCandidate;
    } else if (typeof frame === "object") {
      if (frame.image_name) {
        imagePath = resolveImagePath(`images/${frame.image_name}`);
      }
      mat = frame.pose;
    }
    const c2w = poseToC2w(mat, type);
    if(c2w == null) {
      continue;
    }
    poses.push(c2w);
    legends.push(imagePath ? imagePath.split("/").pop() : String(fid));
    colors.push("#1f77b4");
    imageFiles.push(imagePath ? state.fileMap.get(imagePath) : null);
  }

  return { poses, legends, colors, imageFiles };
}


async function loadNpy() {
  const candidates = listFilesByExtension([".npy", ".npz"]);
  const target = selectNpyCandidate(candidates);
  const buffer = await readBinaryFile(target);
  const payload =
    target.toLowerCase().endsWith(".npz") ? extractNpyFromNpz(buffer) : buffer;
  const parsed = parseNpy(payload, { returnShape: true });
  const poses = posesFromNpyTensor(parsed.data, parsed.shape);
  if (poses.length === 0) {
    throw new Error("NPY tensor contained no valid poses.");
  }
  const baseName = target.split("/").pop() || "pose";
  const legends = poses.map((_, idx) => `${baseName}:${idx}`);
  const colors = poses.map(() => "#1f77b4");
  const imageFiles = poses.map(() => null);
  return { poses, legends, colors, imageFiles };
}


async function loadNerf() {
  const poses = [];
  const legends = [];
  const colors = [];
  const imageFiles = [];

  const json = await readJsonFile("transforms.json");
  const frames = json.frames || [];
  for (let i = 0; i < frames.length; i += 1) {
    const frame = frames[i];
    const c2w = frame.transform_matrix;
    poses.push(ensureMatrix4(c2w));
    colors.push("#1f77b4");

    if (frame.file_path) {
      const cleaned = frame.file_path.replace(/^\.\//, "");
      const imagePath = resolveImagePath(cleaned);
      legends.push(imagePath ? imagePath.split("/").pop() : String(i));
      imageFiles.push(imagePath ? state.fileMap.get(imagePath) : null);
    } else {
      legends.push(String(i));
      imageFiles.push(null);
    }
  }

  return { poses, legends, colors, imageFiles };
}


async function loadColmap() {
  const poses = [];
  const legends = [];
  const colors = [];
  const imageFiles = [];

  const text = await readTextFile("images.txt");
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  let lineIdx = 0;
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith("#")) {
      continue;
    }
    lineIdx += 1;
    if (lineIdx % 2 === 0) {
      continue;
    }
    const elems = line.split(" ");
    const fname = elems.slice(9).join("_");
    legends.push(fname);

    const imagePath = resolveImagePath(`images/${fname}`);
    imageFiles.push(imagePath ? state.fileMap.get(imagePath) : null);

    const qvec = elems.slice(1, 5).map(Number);
    const tvec = elems.slice(5, 8).map(Number);
    const rot = qvecToRotmat([-qvec[0], -qvec[1], -qvec[2], -qvec[3]]);

    const w2c = [
      [rot[0][0], rot[0][1], rot[0][2], tvec[0]],
      [rot[1][0], rot[1][1], rot[1][2], tvec[1]],
      [rot[2][0], rot[2][1], rot[2][2], tvec[2]],
      [0, 0, 0, 1],
    ];
    let c2w = mat4Invert(w2c);

    poses.push(c2w);
    colors.push("#1f77b4");
  }
  return { poses, legends, colors, imageFiles };
}


function poseToC2w(mat, type) {
  if (type === "c2w") {
    return ensureMatrix4(mat);
  }
  if (type === "w2c") {
    const w2c = ensureMatrix4(mat);
    return mat4Invert(w2c);
  }
  if (type === "elu") {
    const matrix = Array.isArray(mat) && Array.isArray(mat[0]) ? mat : asMatrix(mat, 3, 3);
    const eye = matrix[0];
    const lookat = matrix[1];
    const up = matrix[2];
    return eluToC2w(eye, lookat, up);
  }
  if (type === "sph") {
    const vec = ensureVector3(mat);
    const eye = sphericalToCartesian([((vec[0] || 0) * Math.PI) / 180, ((vec[1] || 0) * Math.PI) / 180, vec[2]]);
    return eluToC2w(eye, [0, 0, 0], [0, 0, 1]);
  }
  if (type === "xyz") {
    const eye = ensureVector3(mat);
    return eluToC2w(eye, [0, 0, 0], [0, 0, 1]);
  }
  return null;
}


async function buildPlot() {
  const errors = [];
  setErrors([]);
  setStatus("");

  if (!state.fileMap || state.fileMap.size === 0) {
    errors.push("Please select a scene folder first.");
    setErrors(errors);
    return;
  }

  const format = formatSelect.value;
  const poseType = poseSelect.value;
  const imageSize = Number(imageSizeInput.value || BASE_IMAGE_SIZE);
  const showImages = showImagesInput ? showImagesInput.checked : true;
  const upAxis = upAxisSelect.value || "z-up";
  const recenter = isSwitchOn(recenterInput);
  const hadRendered = state.hasRendered;
  const previousUpAxis = viewer.upAxis;

  let data;
  try {
    if (format === "quick") {
      data = await loadQuick(poseType);
    } else if (format === "nerf") {
      data = await loadNerf();
    } else if (format === "colmap") {
      data = await loadColmap();
    } else if (format === "npy") {
      data = await loadNpy();
    } else {
      throw new Error("Unsupported format.");
    }
  } catch (err) {
    errors.push(err.message || String(err));
    setErrors(errors);
    return;
  }

  let poses = data.poses;
  const legends = data.legends;
  const colors = data.colors;
  const imageFiles = data.imageFiles;
  const coordSystem = getCoordSystem();

  if (coordSystem === "opencv") {
    poses = poses.map((pose) => convertPoseFromOpenCV(pose));
  }

  if (format === "colmap") {
    poses = alignPosesUp(poses);
  }

  if (recenter) {
    poses = recenterCameras(poses);
  }

  let sceneSize = 5.0;
  let rescale = 1.0;
  if (!state.hasRendered) {
    const autoParams = computeAutoScaleAndScene(poses);
    state.autoParams = autoParams;
    rescale = autoParams.scale;
    sceneSize = autoParams.sceneSize;
    rescaleInput.value = rescale.toFixed(4);
    sceneSizeInput.value = sceneSize.toFixed(2);
    state.hasRendered = true;
  } else {
    const rawRescale = rescaleInput.value.trim();
    const rawScene = sceneSizeInput.value.trim();
    const fallbackScale = state.autoParams ? state.autoParams.scale : 1.0;
    const fallbackScene = state.autoParams ? state.autoParams.sceneSize : 5.0;
    if (rawRescale && Number.isNaN(Number(rawRescale))) {
      errors.push("Rescale must be a number.");
      setErrors(errors);
      return;
    }
    if (rawScene && Number.isNaN(Number(rawScene))) {
      errors.push("Scene size must be a number.");
      setErrors(errors);
      return;
    }
    rescale = parseNumberOr(rawRescale, fallbackScale);
    sceneSize = parseNumberOr(rawScene, fallbackScene);
  }

  if (rescale !== 1.0) {
    poses = rescaleCameras(poses, rescale);
  }

  let activeViewer;
  try {
    activeViewer = initViewer(upAxis);
  } catch (err) {
    errors.push(`Three.js init failed: ${err.message || err}`);
    setErrors(errors);
    return;
  }
  const shouldPreserveView =
    hadRendered &&
    previousUpAxis === upAxis &&
    activeViewer.camera &&
    activeViewer.controls;
  const preservedView = shouldPreserveView
    ? {
        position: activeViewer.camera.position.clone(),
        target: activeViewer.controls.target.clone(),
      }
    : null;
  clearContent();
  const showAxes = showAxesInput ? showAxesInput.checked : true;
  const showGrid = showGridInput ? showGridInput.checked : true;
  const helpers = buildHelpers(sceneSize, upAxis, showAxes, showGrid);
  activeViewer.content.add(helpers);
  activeViewer.helpers = helpers.userData ? helpers.userData.helpers : null;

  for (let idx = 0; idx < poses.length; idx += 1) {
    const pose = poses[idx];
    const legend = legends[idx];
    const color = colors[idx] || "#1f77b4";

    if (showImages && imageFiles[idx]) {
      try {
        const imageScale = imageSize > 0 ? imageSize / BASE_IMAGE_SIZE : 1;
        const imageData = await loadImageTexture(imageFiles[idx], imageSize);
        const plane = createImagePlane(
          pose,
          imageData.texture,
          imageData.aspect,
          imageScale
        );
        activeViewer.content.add(plane);
      } catch (err) {
        errors.push(`Image load failed (${legend}): ${err.message || err}`);
      }
    }

    const cone = calcCamConePts3d(pose, 50.0);
    const labelZ = cone[0][2] < 0 ? cone[0][2] - 0.05 : cone[0][2] + 0.05;
    const line = createConeLines(cone, color);
    activeViewer.content.add(line);

    const sprite = createTextSprite(legend, color, 0.4);
    sprite.position.set(cone[0][0], cone[0][1], labelZ);
    activeViewer.content.add(sprite);
  }

  if (errors.length > 0) {
    setErrors(errors);
  }

  updateCameraForScene(sceneSize, upAxis);
  if (preservedView) {
    activeViewer.camera.position.copy(preservedView.position);
    activeViewer.controls.target.copy(preservedView.target);
    activeViewer.controls.update();
  }
  setStatus(`Rendered ${poses.length} camera poses.`);
}


function resetForm() {
  datasetInput.value = "";
  datasetHint.textContent = "Please choose a scene folder.";
  formatSelect.value = "quick";
  poseSelect.value = "auto";
  imageSizeInput.value = 256;
  sceneSizeInput.value = 5;
  rescaleInput.value = "";
  if (showImagesInput) {
    showImagesInput.checked = true;
  }
  upAxisSelect.value = "z-up";
  if (showAxesInput) {
    showAxesInput.checked = true;
  }
  if (showGridInput) {
    showGridInput.checked = true;
  }
  setSwitchState(recenterInput, false);
  state.entries = [];
  state.fileMap = new Map();
  state.root = "";
  state.hasRendered = false;
  state.autoParams = null;
  state.formatAutoDetected = false;
  state.coordSystemTouched = false;
  updatePoseState();
  setErrors([]);
  setStatus("");
  teardownViewer();
}


function isSwitchOn(control) {
  return !!control && control.getAttribute("aria-checked") === "true";
}


function setSwitchState(control, isOn) {
  if (!control) {
    return;
  }
  control.setAttribute("aria-checked", isOn ? "true" : "false");
  control.classList.toggle("is-on", isOn);
}


function getCoordSystem() {
  for (const input of coordSystemInputs) {
    if (input.checked) {
      return input.value;
    }
  }
  return "opengl";
}


function setCoordSystem(value) {
  for (const input of coordSystemInputs) {
    input.checked = input.value === value;
  }
}


function setCoordSystemDefault(format) {
  if (format === "colmap") {
    setCoordSystem("opencv");
    return;
  }
  if (!state.coordSystemTouched) {
    setCoordSystem("opengl");
  }
}


function handleDatasetChange(event) {
  const { entries, root, roots } = normalizeEntriesFromInput(event.target.files || []);
  if (entries.length === 0) {
    resetForm();
    return;
  }
  setDatasetEntries(entries, root, roots);
}


async function handleDirectoryPick() {
  if (!("showDirectoryPicker" in window)) {
    setErrors(["Your browser does not support the File System Access API."]);
    return;
  }
  try {
    const dirHandle = await window.showDirectoryPicker({ mode: "read" });
    const entries = await collectFilesFromHandle(dirHandle);
    if (entries.length === 0) {
      setErrors(["Selected folder has no readable files."]);
      return;
    }
    setDatasetEntries(entries, dirHandle.name);
  } catch (err) {
    if (err && err.name === "AbortError") {
      return;
    }
    if (err && (err.name === "SecurityError" || err.name === "NotAllowedError")) {
      revealLegacyPicker(
        "Directory access was blocked by the browser. Use the legacy picker instead."
      );
      return;
    }
    setErrors([err.message || String(err)]);
  }
}


function attachExampleButtons() {
  document.querySelectorAll(".example").forEach((btn) => {
    btn.addEventListener("click", () => {
      formatSelect.value = btn.dataset.format;
      poseSelect.value = btn.dataset.pose || "auto";
      updatePoseState();
      setStatus("Now pick the matching scene folder with the file picker.");
    });
  });
}


updatePoseState();
attachExampleButtons();

function handleOptionChange() {
  if (!state.fileMap || state.fileMap.size === 0) {
    return;
  }
  buildPlot().catch((err) => setErrors([err.message || String(err)]));
}

formatSelect.addEventListener("change", updatePoseState);
datasetInput.addEventListener("change", handleDatasetChange);
if (chooseFolderButton) {
  const supportsDirPicker = "showDirectoryPicker" in window;
  chooseFolderButton.hidden = !supportsDirPicker;
  if (supportsDirPicker) {
    datasetInput.classList.add("is-hidden");
    chooseFolderButton.addEventListener("click", handleDirectoryPick);
  }
}
if (showAxesInput) {
  showAxesInput.addEventListener("change", updateHelperVisibility);
}
if (showGridInput) {
  showGridInput.addEventListener("change", updateHelperVisibility);
}
if (showImagesInput) {
  showImagesInput.addEventListener("change", handleOptionChange);
}
coordSystemInputs.forEach((input) => {
  input.addEventListener("change", () => {
    state.coordSystemTouched = true;
    handleOptionChange();
  });
});
upAxisSelect.addEventListener("change", handleOptionChange);
if (recenterInput) {
  recenterInput.addEventListener("click", () => {
    setSwitchState(recenterInput, !isSwitchOn(recenterInput));
    handleOptionChange();
  });
}
imageSizeInput.addEventListener("change", handleOptionChange);
sceneSizeInput.addEventListener("change", handleOptionChange);
rescaleInput.addEventListener("change", handleOptionChange);
renderButton.addEventListener("click", () => {
  buildPlot().catch((err) => setErrors([err.message || String(err)]));
});
resetButton.addEventListener("click", resetForm);
