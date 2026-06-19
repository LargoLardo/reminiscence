import React, { useEffect, useRef, useState } from "react";

const SH_C0 = 0.28209479177387814;
const FOCAL = 1.7320508;
const VERTEX_FLOATS = 13;
const POINT_PIXEL_RADIUS = 1.6;
const GAUSSIAN_EXTENT = 2.35;
const MIN_GAUSSIAN_NDC = 0.0012;
const MAX_GAUSSIAN_NDC = 0.28;
const DEFAULT_VIEW = { yaw: 0.5, pitch: -0.18, distance: 1.75, panX: 0, panY: -0.04 };
const CORNERS = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, -1],
  [1, 1],
  [-1, 1],
];
const TYPE_SIZES = {
  char: 1,
  int8: 1,
  uchar: 1,
  uint8: 1,
  short: 2,
  int16: 2,
  ushort: 2,
  uint16: 2,
  int: 4,
  int32: 4,
  uint: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function formatCount(value) {
  return new Intl.NumberFormat().format(value);
}

function percentile(values, amount) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * amount)];
}

function findHeader(bytes) {
  const marker = new TextEncoder().encode("end_header");
  for (let i = 0; i <= bytes.length - marker.length; i += 1) {
    let matched = true;
    for (let j = 0; j < marker.length; j += 1) {
      if (bytes[i + j] !== marker[j]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      let dataOffset = i + marker.length;
      while (bytes[dataOffset] === 10 || bytes[dataOffset] === 13) dataOffset += 1;
      return {
        text: new TextDecoder("ascii").decode(bytes.slice(0, i + marker.length)),
        dataOffset,
      };
    }
  }
  throw new Error("This does not look like a PLY file");
}

function parseHeader(text) {
  let format = "";
  let vertexCount = 0;
  let readingVertex = false;
  const properties = [];

  for (const line of text.split(/\r?\n/).map((entry) => entry.trim())) {
    if (!line || line.startsWith("comment ")) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "format") format = parts[1];
    if (parts[0] === "element") {
      readingVertex = parts[1] === "vertex";
      if (readingVertex) vertexCount = Number(parts[2]);
    }
    if (readingVertex && parts[0] === "property") {
      if (parts[1] === "list") throw new Error("List properties are not supported");
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  if (!vertexCount || !properties.length) throw new Error("PLY file is missing vertex data");
  return { format, vertexCount, properties };
}

function readValue(view, offset, type, littleEndian) {
  switch (type) {
    case "char":
    case "int8":
      return view.getInt8(offset);
    case "uchar":
    case "uint8":
      return view.getUint8(offset);
    case "short":
    case "int16":
      return view.getInt16(offset, littleEndian);
    case "ushort":
    case "uint16":
      return view.getUint16(offset, littleEndian);
    case "int":
    case "int32":
      return view.getInt32(offset, littleEndian);
    case "uint":
    case "uint32":
      return view.getUint32(offset, littleEndian);
    case "float":
    case "float32":
      return view.getFloat32(offset, littleEndian);
    case "double":
    case "float64":
      return view.getFloat64(offset, littleEndian);
    default:
      throw new Error(`Unsupported PLY property type: ${type}`);
  }
}

function colorFrom(values) {
  let rgb = [0.94, 0.78, 0.62];
  if (values.f_dc_0 != null && values.f_dc_1 != null && values.f_dc_2 != null) {
    rgb = [
      clamp(0.5 + SH_C0 * values.f_dc_0, 0, 1),
      clamp(0.5 + SH_C0 * values.f_dc_1, 0, 1),
      clamp(0.5 + SH_C0 * values.f_dc_2, 0, 1),
    ];
  } else if (values.red != null && values.green != null && values.blue != null) {
    rgb = [values.red / 255, values.green / 255, values.blue / 255];
  }
  return rgb.map((channel) => Math.pow(clamp(channel * 1.08, 0, 1), 1 / 2.2));
}

function scaleFrom(values) {
  if (values.scale_0 == null || values.scale_1 == null || values.scale_2 == null) {
    return [0.035, 0.035, 0.035];
  }
  return [Math.exp(values.scale_0), Math.exp(values.scale_1), Math.exp(values.scale_2)];
}

function rotationFrom(values) {
  const quat = [values.rot_0 ?? 1, values.rot_1 ?? 0, values.rot_2 ?? 0, values.rot_3 ?? 0];
  const length = Math.hypot(...quat) || 1;
  return quat.map((value) => value / length);
}

function axesFromQuaternion(rotation, scales) {
  const [w, x, y, z] = rotation;
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;
  return [
    (1 - yy - zz) * scales[0],
    (xy + wz) * scales[0],
    (xz - wy) * scales[0],
    (xy - wz) * scales[1],
    (1 - xx - zz) * scales[1],
    (yz + wx) * scales[1],
    (xz + wy) * scales[2],
    (yz - wx) * scales[2],
    (1 - xx - yy) * scales[2],
  ];
}

function normalizeCloud(rawPositions, colors, opacities, rawScales, rawRotations, vertexCount) {
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < rawPositions.length; i += 3) {
    xs.push(rawPositions[i]);
    ys.push(rawPositions[i + 1]);
    zs.push(rawPositions[i + 2]);
  }

  const low = [percentile(xs, 0.05), percentile(ys, 0.05), percentile(zs, 0.05)];
  const high = [percentile(xs, 0.95), percentile(ys, 0.95), percentile(zs, 0.95)];
  const size = high.map((value, index) => Math.max(value - low[index], 0.0001));
  const center = high.map((value, index) => (value + low[index]) / 2);
  const sceneScale = 2.35 / Math.max(...size);
  const positions = new Float32Array(rawPositions.length);
  const axes = new Float32Array(vertexCount * 9);
  const focusRadii = [];

  for (let i = 0; i < rawPositions.length; i += 3) {
    const x = (rawPositions[i] - center[0]) * sceneScale;
    const y = (rawPositions[i + 1] - center[1]) * sceneScale;
    const z = (rawPositions[i + 2] - center[2]) * sceneScale;
    positions[i] = x;
    positions[i + 1] = y;
    positions[i + 2] = z;
    focusRadii.push(Math.hypot(x, y, z));
  }

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const scaleIndex = vertex * 3;
    const rotationIndex = vertex * 4;
    const scaledAxes = axesFromQuaternion(
      [
        rawRotations[rotationIndex],
        rawRotations[rotationIndex + 1],
        rawRotations[rotationIndex + 2],
        rawRotations[rotationIndex + 3],
      ],
      [
        clamp(rawScales[scaleIndex] * sceneScale, 0.002, 0.22),
        clamp(rawScales[scaleIndex + 1] * sceneScale, 0.002, 0.22),
        clamp(rawScales[scaleIndex + 2] * sceneScale, 0.002, 0.22),
      ],
    );
    axes.set(scaledAxes, vertex * 9);
  }

  return {
    positions,
    colors,
    opacities,
    axes,
    vertexCount,
    fitDistance: clamp(percentile(focusRadii, 0.82) * 2.15, 1.55, 3.35),
  };
}

function readVertex(values, vertex, rawPositions, colors, opacities, rawScales, rawRotations) {
  const pointIndex = vertex * 3;
  rawPositions[pointIndex] = values.x ?? 0;
  rawPositions[pointIndex + 1] = values.y ?? 0;
  rawPositions[pointIndex + 2] = values.z ?? 0;

  colors.set(colorFrom(values), pointIndex);
  opacities[vertex] = values.opacity == null ? 0.86 : clamp(sigmoid(values.opacity), 0.02, 0.98);
  rawScales.set(scaleFrom(values), pointIndex);
  rawRotations.set(rotationFrom(values), vertex * 4);
}

function parseBinaryPly(buffer, dataOffset, vertexCount, properties, littleEndian) {
  const view = new DataView(buffer);
  let stride = 0;
  const layout = properties.map((property) => {
    const size = TYPE_SIZES[property.type];
    if (!size) throw new Error(`Unsupported PLY property type: ${property.type}`);
    const entry = { ...property, offset: stride };
    stride += size;
    return entry;
  });
  if (dataOffset + stride * vertexCount > buffer.byteLength) {
    throw new Error("PLY file ended before all vertices could be read");
  }

  const rawPositions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const rawScales = new Float32Array(vertexCount * 3);
  const rawRotations = new Float32Array(vertexCount * 4);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const base = dataOffset + vertex * stride;
    const values = {};
    for (const property of layout) {
      values[property.name] = readValue(view, base + property.offset, property.type, littleEndian);
    }
    readVertex(values, vertex, rawPositions, colors, opacities, rawScales, rawRotations);
  }

  return normalizeCloud(rawPositions, colors, opacities, rawScales, rawRotations, vertexCount);
}

function parseAsciiPly(bytes, dataOffset, vertexCount, properties) {
  const lines = new TextDecoder("ascii").decode(bytes.slice(dataOffset)).split(/\r?\n/);
  const rawPositions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const rawScales = new Float32Array(vertexCount * 3);
  const rawRotations = new Float32Array(vertexCount * 4);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const parts = (lines[vertex] || "").trim().split(/\s+/).map(Number);
    const values = {};
    properties.forEach((property, index) => {
      values[property.name] = parts[index];
    });
    readVertex(values, vertex, rawPositions, colors, opacities, rawScales, rawRotations);
  }

  return normalizeCloud(rawPositions, colors, opacities, rawScales, rawRotations, vertexCount);
}

function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const { text, dataOffset } = findHeader(bytes);
  const { format, vertexCount, properties } = parseHeader(text);
  if (format === "binary_little_endian") {
    return parseBinaryPly(buffer, dataOffset, vertexCount, properties, true);
  }
  if (format === "binary_big_endian") {
    return parseBinaryPly(buffer, dataOffset, vertexCount, properties, false);
  }
  if (format === "ascii") return parseAsciiPly(bytes, dataOffset, vertexCount, properties);
  throw new Error(`Unsupported PLY format: ${format || "unknown"}`);
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message || "Could not compile WebGL shader");
  }
  return shader;
}

function createProgram(gl) {
  const vertexShader = compileShader(
    gl,
    gl.VERTEX_SHADER,
    `
      attribute vec3 a_center;
      attribute vec3 a_color;
      attribute float a_opacity;
      attribute vec2 a_axis1;
      attribute vec2 a_axis2;
      attribute vec2 a_corner;

      varying vec3 v_color;
      varying float v_opacity;
      varying vec2 v_corner;

      void main() {
        vec2 offset = a_axis1 * a_corner.x + a_axis2 * a_corner.y;
        gl_Position = vec4(a_center.xy + offset, a_center.z, 1.0);
        v_color = a_color;
        v_opacity = a_opacity;
        v_corner = a_corner;
      }
    `,
  );
  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;

      varying vec3 v_color;
      varying float v_opacity;
      varying vec2 v_corner;

      void main() {
        float radiusSquared = dot(v_corner, v_corner);
        if (radiusSquared > 1.0) discard;
        float alpha = clamp(v_opacity * exp(-radiusSquared * 2.6) * 1.35, 0.0, 0.98);
        if (alpha < 0.01) discard;
        gl_FragColor = vec4(v_color * alpha, alpha);
      }
    `,
  );
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || "Could not link WebGL program");
  }
  return program;
}

function rotatePoint(position, view) {
  const yawCos = Math.cos(view.yaw);
  const yawSin = Math.sin(view.yaw);
  const pitchCos = Math.cos(view.pitch);
  const pitchSin = Math.sin(view.pitch);
  const yawX = yawCos * position[0] + yawSin * position[2];
  const yawZ = -yawSin * position[0] + yawCos * position[2];
  return [
    yawX + view.panX,
    pitchCos * position[1] - pitchSin * yawZ + view.panY,
    pitchSin * position[1] + pitchCos * yawZ - view.distance,
  ];
}

function rotateVector(vector, view) {
  const yawCos = Math.cos(view.yaw);
  const yawSin = Math.sin(view.yaw);
  const pitchCos = Math.cos(view.pitch);
  const pitchSin = Math.sin(view.pitch);
  const yawX = yawCos * vector[0] + yawSin * vector[2];
  const yawZ = -yawSin * vector[0] + yawCos * vector[2];
  return [yawX, pitchCos * vector[1] - pitchSin * yawZ, pitchSin * vector[1] + pitchCos * yawZ];
}

function projectPoint(point, aspect) {
  return [(point[0] * FOCAL) / (aspect * -point[2]), (point[1] * FOCAL) / -point[2]];
}

function projectDelta(point, vector, aspect) {
  const z2 = point[2] * point[2];
  return [
    (-FOCAL / aspect) * ((vector[0] * point[2] - point[0] * vector[2]) / z2),
    -FOCAL * ((vector[1] * point[2] - point[1] * vector[2]) / z2),
  ];
}

function clampAxis(axis) {
  const length = Math.hypot(axis[0], axis[1]);
  if (!length) return [MIN_GAUSSIAN_NDC, 0];
  const next = clamp(length, MIN_GAUSSIAN_NDC, MAX_GAUSSIAN_NDC);
  return [(axis[0] / length) * next, (axis[1] / length) * next];
}

function gaussianAxes(pointCloud, index, centerCamera, aspect, view) {
  const base = index * 9;
  let xx = 0;
  let xy = 0;
  let yy = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const offset = base + axis * 3;
    const projected = projectDelta(
      centerCamera,
      rotateVector(
        [
          pointCloud.axes[offset],
          pointCloud.axes[offset + 1],
          pointCloud.axes[offset + 2],
        ],
        view,
      ),
      aspect,
    );
    xx += projected[0] * projected[0];
    xy += projected[0] * projected[1];
    yy += projected[1] * projected[1];
  }

  const trace = (xx + yy) * 0.5;
  const diff = (xx - yy) * 0.5;
  const radius = Math.sqrt(diff * diff + xy * xy);
  const lambda1 = Math.max(trace + radius, 0.0);
  const lambda2 = Math.max(trace - radius, 0.0);
  let axis1 = Math.abs(xy) > 1e-8 ? [xy, lambda1 - xx] : xx >= yy ? [1, 0] : [0, 1];
  const axisLength = Math.hypot(axis1[0], axis1[1]) || 1;
  axis1 = [axis1[0] / axisLength, axis1[1] / axisLength];
  return [
    clampAxis([axis1[0] * Math.sqrt(lambda1) * GAUSSIAN_EXTENT, axis1[1] * Math.sqrt(lambda1) * GAUSSIAN_EXTENT]),
    clampAxis([-axis1[1] * Math.sqrt(lambda2) * GAUSSIAN_EXTENT, axis1[0] * Math.sqrt(lambda2) * GAUSSIAN_EXTENT]),
  ];
}

function createInitialView(pointCloud) {
  return { ...DEFAULT_VIEW, distance: pointCloud?.fitDistance ?? DEFAULT_VIEW.distance };
}

function createRenderer(canvas, pointCloud, mode) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false, premultipliedAlpha: true });
  if (!gl) throw new Error("WebGL is not available in this browser");

  const program = createProgram(gl);
  const buffer = gl.createBuffer();
  const vertexData = new Float32Array(pointCloud.vertexCount * CORNERS.length * VERTEX_FLOATS);
  const depths = new Float32Array(pointCloud.vertexCount);
  const order = Array.from({ length: pointCloud.vertexCount }, (_, index) => index);
  const locations = {
    center: gl.getAttribLocation(program, "a_center"),
    color: gl.getAttribLocation(program, "a_color"),
    opacity: gl.getAttribLocation(program, "a_opacity"),
    axis1: gl.getAttribLocation(program, "a_axis1"),
    axis2: gl.getAttribLocation(program, "a_axis2"),
    corner: gl.getAttribLocation(program, "a_corner"),
  };

  function writeQuad(offset, center, colorIndex, opacity, axis1, axis2) {
    for (const corner of CORNERS) {
      vertexData[offset] = center[0];
      vertexData[offset + 1] = center[1];
      vertexData[offset + 2] = 0;
      vertexData[offset + 3] = pointCloud.colors[colorIndex];
      vertexData[offset + 4] = pointCloud.colors[colorIndex + 1];
      vertexData[offset + 5] = pointCloud.colors[colorIndex + 2];
      vertexData[offset + 6] = opacity;
      vertexData[offset + 7] = axis1[0];
      vertexData[offset + 8] = axis1[1];
      vertexData[offset + 9] = axis2[0];
      vertexData[offset + 10] = axis2[1];
      vertexData[offset + 11] = corner[0];
      vertexData[offset + 12] = corner[1];
      offset += VERTEX_FLOATS;
    }
    return offset;
  }

  function draw(view) {
    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const aspect = width / height;
    for (let i = 0; i < pointCloud.vertexCount; i += 1) {
      const p = rotatePoint(
        [pointCloud.positions[i * 3], pointCloud.positions[i * 3 + 1], pointCloud.positions[i * 3 + 2]],
        view,
      );
      depths[i] = p[2] < -0.02 ? -p[2] : -1;
    }
    order.sort((a, b) => depths[b] - depths[a]);

    let offset = 0;
    let rendered = 0;
    const pointAxes = [
      [(POINT_PIXEL_RADIUS * 2) / width, 0],
      [0, (POINT_PIXEL_RADIUS * 2) / height],
    ];
    for (const index of order) {
      if (depths[index] <= 0) continue;
      const colorIndex = index * 3;
      const centerCamera = rotatePoint(
        [pointCloud.positions[colorIndex], pointCloud.positions[colorIndex + 1], pointCloud.positions[colorIndex + 2]],
        view,
      );
      const center = projectPoint(centerCamera, aspect);
      if (Math.abs(center[0]) > 2.5 || Math.abs(center[1]) > 2.5) continue;

      const axes = mode === "point" ? pointAxes : gaussianAxes(pointCloud, index, centerCamera, aspect, view);
      const opacity = mode === "point" ? Math.min(pointCloud.opacities[index] * 1.25, 0.95) : pointCloud.opacities[index];
      offset = writeQuad(offset, center, colorIndex, opacity, axes[0], axes[1]);
      rendered += CORNERS.length;
    }

    gl.viewport(0, 0, width, height);
    gl.clearColor(0.055, 0.045, 0.038, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData.subarray(0, offset), gl.DYNAMIC_DRAW);

    const stride = VERTEX_FLOATS * 4;
    gl.enableVertexAttribArray(locations.center);
    gl.vertexAttribPointer(locations.center, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(locations.color);
    gl.vertexAttribPointer(locations.color, 3, gl.FLOAT, false, stride, 3 * 4);
    gl.enableVertexAttribArray(locations.opacity);
    gl.vertexAttribPointer(locations.opacity, 1, gl.FLOAT, false, stride, 6 * 4);
    gl.enableVertexAttribArray(locations.axis1);
    gl.vertexAttribPointer(locations.axis1, 2, gl.FLOAT, false, stride, 7 * 4);
    gl.enableVertexAttribArray(locations.axis2);
    gl.vertexAttribPointer(locations.axis2, 2, gl.FLOAT, false, stride, 9 * 4);
    gl.enableVertexAttribArray(locations.corner);
    gl.vertexAttribPointer(locations.corner, 2, gl.FLOAT, false, stride, 11 * 4);
    gl.drawArrays(gl.TRIANGLES, 0, rendered);
  }

  function dispose() {
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
  }

  return { draw, dispose };
}

function ViewerPane({ label, canvasRef, status, error, bindPointerHandlers }) {
  return (
    <article className="viewer-card">
      <div className="viewer-title">
        <p className="eyebrow">{label}</p>
      </div>
      <div className="viewer-canvas-wrap" {...bindPointerHandlers()}>
        <canvas ref={canvasRef} aria-label={label} />
        {status !== "ready" && (
          <div className="viewer-overlay">
            {status === "loading" ? "Loading Gaussian splats..." : error}
          </div>
        )}
      </div>
    </article>
  );
}

export default function PlyViewer({ src }) {
  const pointCanvasRef = useRef(null);
  const gaussianCanvasRef = useRef(null);
  const fullscreenCanvasRef = useRef(null);
  const fullscreenShellRef = useRef(null);
  const renderersRef = useRef([]);
  const fullscreenRendererRef = useRef(null);
  const dragRef = useRef(null);
  const keysRef = useRef(new Set());
  const [pointCloud, setPointCloud] = useState(null);
  const [view, setView] = useState(DEFAULT_VIEW);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function load() {
      setStatus("loading");
      setError("");
      setPointCloud(null);
      try {
        const response = await fetch(src, { signal: controller.signal });
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(body?.detail || "Could not load the PLY file");
        }
        const cloud = parsePly(await response.arrayBuffer());
        if (!cancelled) {
          setPointCloud(cloud);
          setView(createInitialView(cloud));
          setStatus("ready");
        }
      } catch (loadError) {
        if (!cancelled && loadError.name !== "AbortError") {
          setStatus("failed");
          setError(loadError.message);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [src]);

  useEffect(() => {
    if (!pointCloud || !pointCanvasRef.current || !gaussianCanvasRef.current) return undefined;
    try {
      const renderers = [
        createRenderer(pointCanvasRef.current, pointCloud, "point"),
        createRenderer(gaussianCanvasRef.current, pointCloud, "gaussian"),
      ];
      const observers = [pointCanvasRef.current, gaussianCanvasRef.current].map((canvas, index) => {
        const observer = new ResizeObserver(() => renderers[index].draw(view));
        observer.observe(canvas);
        return observer;
      });
      renderers.forEach((renderer) => renderer.draw(view));
      renderersRef.current = renderers;
      return () => {
        observers.forEach((observer) => observer.disconnect());
        renderers.forEach((renderer) => renderer.dispose());
        renderersRef.current = [];
      };
    } catch (rendererError) {
      setStatus("failed");
      setError(rendererError.message);
      return undefined;
    }
  }, [pointCloud]);

  useEffect(() => {
    renderersRef.current.forEach((renderer) => renderer.draw(view));
    fullscreenRendererRef.current?.draw(view);
  }, [view]);

  useEffect(() => {
    if (!fullscreen || !pointCloud || !fullscreenCanvasRef.current) return undefined;

    const renderer = createRenderer(fullscreenCanvasRef.current, pointCloud, "gaussian");
    const observer = new ResizeObserver(() => renderer.draw(view));
    observer.observe(fullscreenCanvasRef.current);
    renderer.draw(view);
    fullscreenRendererRef.current = renderer;

    return () => {
      observer.disconnect();
      renderer.dispose();
      fullscreenRendererRef.current = null;
    };
  }, [fullscreen, pointCloud]);

  useEffect(() => {
    if (!fullscreen) return undefined;

    const shell = fullscreenShellRef.current;
    shell?.focus();
    shell?.requestFullscreen?.().catch(() => {});

    function handleFullscreenChange() {
      if (!document.fullscreenElement) setFullscreen(false);
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [fullscreen]);

  useEffect(() => {
    if (!fullscreen) return undefined;

    const keys = keysRef.current;
    keys.clear();
    const trackedKeys = new Set(["w", "a", "s", "d", "shift", "control"]);

    function keyName(event) {
      return event.key === " " ? "space" : event.key.toLowerCase();
    }

    function handleKeyDown(event) {
      const key = keyName(event);
      if (key === "escape") {
        closeFullscreen();
        return;
      }
      if (trackedKeys.has(key)) {
        event.preventDefault();
        keys.add(key);
      }
    }

    function handleKeyUp(event) {
      keys.delete(keyName(event));
    }

    let frame = 0;
    function move() {
      setView((current) => {
        const forward = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
        const strafe = (keys.has("d") ? 1 : 0) - (keys.has("a") ? 1 : 0);
        const vertical = (keys.has("shift") ? 1 : 0) - (keys.has("control") ? 1 : 0);
        if (!forward && !strafe && !vertical) return current;

        const speed = 0.026 * Math.max(current.distance, 1);
        return {
          ...current,
          distance: clamp(current.distance - forward * speed, 0.25, 9),
          panX: current.panX + strafe * speed * 0.62,
          panY: current.panY + vertical * speed * 0.62,
        };
      });
      frame = requestAnimationFrame(move);
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    frame = requestAnimationFrame(move);

    return () => {
      cancelAnimationFrame(frame);
      keys.clear();
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [fullscreen]);

  function bindPointerHandlers() {
    return {
      onPointerDown(event) {
        event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = {
          x: event.clientX,
          y: event.clientY,
          mode: fullscreen ? "rotate" : event.shiftKey || event.altKey ? "pan" : "rotate",
          view,
        };
      },
      onPointerMove(event) {
        if (!dragRef.current) return;
        const drag = dragRef.current;
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        if (drag.mode === "pan") {
          const panSpeed = 0.0022 * drag.view.distance;
          setView({ ...drag.view, panX: drag.view.panX + dx * panSpeed, panY: drag.view.panY - dy * panSpeed });
        } else {
          setView({ ...drag.view, yaw: drag.view.yaw + dx * 0.008, pitch: clamp(drag.view.pitch + dy * 0.008, -1.45, 1.45) });
        }
      },
      onPointerUp(event) {
        if (dragRef.current) event.currentTarget.releasePointerCapture(event.pointerId);
        dragRef.current = null;
      },
      onPointerCancel(event) {
        if (dragRef.current) event.currentTarget.releasePointerCapture(event.pointerId);
        dragRef.current = null;
      },
      onDoubleClick() {
        setView(createInitialView(pointCloud));
      },
      onWheel(event) {
        event.preventDefault();
        setView((current) => ({ ...current, distance: clamp(current.distance * Math.exp(event.deltaY * 0.001), 0.5, 8) }));
      },
    };
  }

  function openFullscreen() {
    if (pointCloud) setFullscreen(true);
  }

  function closeFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen?.().catch(() => {});
    }
    setFullscreen(false);
  }

  return (
    <div className="viewer-stack">
      <div className="viewer-duo">
        <ViewerPane
          label="GAUSSIAN SPLAT"
          canvasRef={gaussianCanvasRef}
          status={status}
          error={error}
          bindPointerHandlers={bindPointerHandlers}
        />
        <ViewerPane
          label="POINT CLOUD"
          canvasRef={pointCanvasRef}
          status={status}
          error={error}
          bindPointerHandlers={bindPointerHandlers}
        />
      </div>
      <div className="viewer-toolbar">
        <p className="viewer-meta">
          {pointCloud ? `${formatCount(pointCloud.vertexCount)} Gaussian splats loaded` : "Preparing splat renderer"}
        </p>
        <button type="button" className="ghost-button" onClick={() => setView(createInitialView(pointCloud))}>
          Reset view
        </button>
        <button type="button" className="ghost-button" disabled={!pointCloud} onClick={openFullscreen}>
          Fullscreen
        </button>
      </div>
      {fullscreen && (
        <div ref={fullscreenShellRef} className="fullscreen-viewer" tabIndex={-1}>
          <div className="fullscreen-bar">
            <p className="viewer-meta">WASD move. Shift up. Ctrl down. Drag to look.</p>
            <button type="button" className="ghost-button" onClick={closeFullscreen}>
              Exit
            </button>
          </div>
          <div className="fullscreen-canvas-wrap" {...bindPointerHandlers()}>
            <canvas ref={fullscreenCanvasRef} aria-label="Fullscreen Gaussian splat viewer" />
          </div>
        </div>
      )}
    </div>
  );
}
