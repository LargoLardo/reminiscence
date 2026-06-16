import React, { useEffect, useRef, useState } from "react";

const SH_C0 = 0.28209479177387814;
const DEFAULT_VIEW = {
  yaw: 0.5,
  pitch: -0.18,
  distance: 1.75,
  panX: 0,
  panY: 0.04,
};
const SPLAT_SCALE_BOOST = 0.95;
const MIN_SPLAT_RADIUS = 0.007;
const MAX_SPLAT_RADIUS = 0.14;
const VERTEX_FLOATS = 11;
const SPLAT_CORNERS = [
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

function percentile(values, percentileValue) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * percentileValue)];
}

function findHeader(array) {
  const marker = new TextEncoder().encode("end_header");
  for (let i = 0; i <= array.length - marker.length; i += 1) {
    let matched = true;
    for (let j = 0; j < marker.length; j += 1) {
      if (array[i + j] !== marker[j]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      let dataOffset = i + marker.length;
      while (array[dataOffset] === 10 || array[dataOffset] === 13) dataOffset += 1;
      return {
        text: new TextDecoder("ascii").decode(array.slice(0, i + marker.length)),
        dataOffset,
      };
    }
  }
  throw new Error("This does not look like a PLY file");
}

function parseHeader(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim());
  let format = "";
  let vertexCount = 0;
  let readingVertexProperties = false;
  const properties = [];

  for (const line of lines) {
    if (!line || line.startsWith("comment ")) continue;

    const parts = line.split(/\s+/);
    if (parts[0] === "format") {
      format = parts[1];
      continue;
    }

    if (parts[0] === "element") {
      readingVertexProperties = parts[1] === "vertex";
      if (readingVertexProperties) vertexCount = Number(parts[2]);
      continue;
    }

    if (readingVertexProperties && parts[0] === "property") {
      if (parts[1] === "list") {
        throw new Error("List properties are not supported in the browser viewer yet");
      }
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  if (!vertexCount || !properties.length) {
    throw new Error("PLY file is missing vertex data");
  }

  return { format, vertexCount, properties };
}

function readValue(view, offset, type, littleEndian = true) {
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

function getColor(values) {
  let color = [0.94, 0.78, 0.62];

  if (values.f_dc_0 != null && values.f_dc_1 != null && values.f_dc_2 != null) {
    color = [
      clamp(0.5 + SH_C0 * values.f_dc_0, 0, 1),
      clamp(0.5 + SH_C0 * values.f_dc_1, 0, 1),
      clamp(0.5 + SH_C0 * values.f_dc_2, 0, 1),
    ];
  } else if (values.red != null && values.green != null && values.blue != null) {
    color = [
      clamp(values.red / 255, 0, 1),
      clamp(values.green / 255, 0, 1),
      clamp(values.blue / 255, 0, 1),
    ];
  }

  return color.map((channel) => Math.pow(clamp(channel * 1.08, 0, 1), 1 / 2.2));
}

function getOpacity(values) {
  if (values.opacity == null) return 0.85;
  return clamp(sigmoid(values.opacity), 0.015, 0.985);
}

function getRawScale(values) {
  if (
    values.scale_0 == null
    || values.scale_1 == null
    || values.scale_2 == null
  ) {
    return [0.08, 0.08];
  }

  const axes = [
    Math.exp(values.scale_0),
    Math.exp(values.scale_1),
    Math.exp(values.scale_2),
  ].sort((a, b) => b - a);

  return [axes[0], axes[1]];
}

function normalizeSplats(rawPositions, colors, opacities, rawScales, vertexCount) {
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
  const size = [
    Math.max(high[0] - low[0], 0.0001),
    Math.max(high[1] - low[1], 0.0001),
    Math.max(high[2] - low[2], 0.0001),
  ];
  const largestAxis = Math.max(size[0], size[1], size[2], 0.0001);
  const center = [
    (low[0] + high[0]) / 2,
    (low[1] + high[1]) / 2,
    (low[2] + high[2]) / 2,
  ];
  const sceneScale = 2.35 / largestAxis;
  const positions = new Float32Array(rawPositions.length);
  const scales = new Float32Array(vertexCount * 2);
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

  for (let i = 0; i < vertexCount; i += 1) {
    const scaleIndex = i * 2;
    const sx = rawScales[scaleIndex] * sceneScale * SPLAT_SCALE_BOOST;
    const sy = rawScales[scaleIndex + 1] * sceneScale * SPLAT_SCALE_BOOST;

    scales[scaleIndex] = clamp(sx, MIN_SPLAT_RADIUS, MAX_SPLAT_RADIUS);
    scales[scaleIndex + 1] = clamp(sy, MIN_SPLAT_RADIUS, MAX_SPLAT_RADIUS);
  }

  const focusRadius = percentile(focusRadii, 0.82);
  const fitDistance = clamp(focusRadius * 1.65, 1.25, 2.6);

  return {
    positions,
    colors,
    opacities,
    scales,
    vertexCount,
    fitDistance,
    bounds: { width: size[0], height: size[1], depth: size[2] },
  };
}

function parseBinaryPly(buffer, dataOffset, vertexCount, properties, littleEndian) {
  const view = new DataView(buffer);
  let stride = 0;
  const layout = properties.map((property) => {
    const size = TYPE_SIZES[property.type];
    if (!size) throw new Error(`Unsupported PLY property type: ${property.type}`);
    const entry = { ...property, offset: stride, size };
    stride += size;
    return entry;
  });

  if (dataOffset + stride * vertexCount > buffer.byteLength) {
    throw new Error("PLY file ended before all vertices could be read");
  }

  const rawPositions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const rawScales = new Float32Array(vertexCount * 2);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const base = dataOffset + vertex * stride;
    const values = {};
    for (const property of layout) {
      values[property.name] = readValue(
        view,
        base + property.offset,
        property.type,
        littleEndian,
      );
    }

    const pointIndex = vertex * 3;
    rawPositions[pointIndex] = values.x ?? 0;
    rawPositions[pointIndex + 1] = values.y ?? 0;
    rawPositions[pointIndex + 2] = values.z ?? 0;

    const color = getColor(values);
    colors[pointIndex] = color[0];
    colors[pointIndex + 1] = color[1];
    colors[pointIndex + 2] = color[2];

    opacities[vertex] = getOpacity(values);

    const scale = getRawScale(values);
    rawScales[vertex * 2] = scale[0];
    rawScales[vertex * 2 + 1] = scale[1];
  }

  return normalizeSplats(rawPositions, colors, opacities, rawScales, vertexCount);
}

function parseAsciiPly(bytes, dataOffset, vertexCount, properties) {
  const body = new TextDecoder("ascii").decode(bytes.slice(dataOffset));
  const lines = body.split(/\r?\n/).filter(Boolean);
  const rawPositions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const rawScales = new Float32Array(vertexCount * 2);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const parts = lines[vertex].trim().split(/\s+/).map(Number);
    const values = {};
    properties.forEach((property, index) => {
      values[property.name] = parts[index];
    });

    const pointIndex = vertex * 3;
    rawPositions[pointIndex] = values.x ?? 0;
    rawPositions[pointIndex + 1] = values.y ?? 0;
    rawPositions[pointIndex + 2] = values.z ?? 0;

    const color = getColor(values);
    colors[pointIndex] = color[0];
    colors[pointIndex + 1] = color[1];
    colors[pointIndex + 2] = color[2];

    opacities[vertex] = getOpacity(values);

    const scale = getRawScale(values);
    rawScales[vertex * 2] = scale[0];
    rawScales[vertex * 2 + 1] = scale[1];
  }

  return normalizeSplats(rawPositions, colors, opacities, rawScales, vertexCount);
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

  if (format === "ascii") {
    return parseAsciiPly(bytes, dataOffset, vertexCount, properties);
  }

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
      attribute vec3 a_position;
      attribute vec3 a_color;
      attribute float a_opacity;
      attribute vec2 a_scale;
      attribute vec2 a_corner;

      uniform vec2 u_rotation;
      uniform vec2 u_pan;
      uniform float u_distance;
      uniform float u_aspect;

      varying vec3 v_color;
      varying float v_opacity;
      varying vec2 v_corner;

      void main() {
        vec3 point = a_position;

        float yawCos = cos(u_rotation.x);
        float yawSin = sin(u_rotation.x);
        point = vec3(
          yawCos * point.x + yawSin * point.z,
          point.y,
          -yawSin * point.x + yawCos * point.z
        );

        float pitchCos = cos(u_rotation.y);
        float pitchSin = sin(u_rotation.y);
        point = vec3(
          point.x,
          pitchCos * point.y - pitchSin * point.z,
          pitchSin * point.y + pitchCos * point.z
        );

        point.xy += u_pan;
        point.z -= u_distance;

        float fov = 1.7320508;
        float near = 0.01;
        float far = 100.0;
        float clipZ = ((far + near) / (near - far)) * point.z
          + (2.0 * far * near) / (near - far);
        float clipW = max(0.01, -point.z);
        vec4 clip = vec4(
          point.x * fov / u_aspect,
          point.y * fov,
          clipZ,
          clipW
        );

        clip.xy += vec2(
          a_corner.x * a_scale.x * fov / u_aspect,
          a_corner.y * a_scale.y * fov
        );

        gl_Position = clip;
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

        float gaussian = exp(-radiusSquared * 3.1);
        float feather = 1.0 - smoothstep(0.72, 1.0, radiusSquared);
        float alpha = clamp(v_opacity * gaussian * feather * 1.15, 0.0, 0.96);
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

function createInitialView(pointCloud) {
  return {
    ...DEFAULT_VIEW,
    distance: pointCloud?.fitDistance ?? DEFAULT_VIEW.distance,
  };
}

function cameraDepth(pointCloud, index, view) {
  const positionIndex = index * 3;
  const x = pointCloud.positions[positionIndex];
  const y = pointCloud.positions[positionIndex + 1];
  const z = pointCloud.positions[positionIndex + 2];
  const yawCos = Math.cos(view.yaw);
  const yawSin = Math.sin(view.yaw);
  const pitchCos = Math.cos(view.pitch);
  const pitchSin = Math.sin(view.pitch);
  const yawZ = -yawSin * x + yawCos * z;
  const rotatedY = pitchCos * y - pitchSin * yawZ;
  const rotatedZ = pitchSin * y + pitchCos * yawZ;

  return view.distance - rotatedZ + rotatedY * 0.0001;
}

function createRenderer(canvas, pointCloud) {
  const gl = canvas.getContext("webgl", {
    antialias: true,
    alpha: false,
    premultipliedAlpha: true,
  });
  if (!gl) throw new Error("WebGL is not available in this browser");

  const program = createProgram(gl);
  const vertexBuffer = gl.createBuffer();
  const maxVertexCount = pointCloud.vertexCount * SPLAT_CORNERS.length;
  const vertexData = new Float32Array(maxVertexCount * VERTEX_FLOATS);
  const order = Array.from({ length: pointCloud.vertexCount }, (_, index) => index);
  const depths = new Float32Array(pointCloud.vertexCount);
  const locations = {
    position: gl.getAttribLocation(program, "a_position"),
    color: gl.getAttribLocation(program, "a_color"),
    opacity: gl.getAttribLocation(program, "a_opacity"),
    scale: gl.getAttribLocation(program, "a_scale"),
    corner: gl.getAttribLocation(program, "a_corner"),
    rotation: gl.getUniformLocation(program, "u_rotation"),
    pan: gl.getUniformLocation(program, "u_pan"),
    distance: gl.getUniformLocation(program, "u_distance"),
    aspect: gl.getUniformLocation(program, "u_aspect"),
  };

  function writeSplat(dataOffset, splatIndex) {
    const positionIndex = splatIndex * 3;
    const scaleIndex = splatIndex * 2;
    const x = pointCloud.positions[positionIndex];
    const y = pointCloud.positions[positionIndex + 1];
    const z = pointCloud.positions[positionIndex + 2];
    const red = pointCloud.colors[positionIndex];
    const green = pointCloud.colors[positionIndex + 1];
    const blue = pointCloud.colors[positionIndex + 2];
    const opacity = pointCloud.opacities[splatIndex];
    const scaleX = pointCloud.scales[scaleIndex];
    const scaleY = pointCloud.scales[scaleIndex + 1];
    let offset = dataOffset;

    for (const corner of SPLAT_CORNERS) {
      vertexData[offset] = x;
      vertexData[offset + 1] = y;
      vertexData[offset + 2] = z;
      vertexData[offset + 3] = red;
      vertexData[offset + 4] = green;
      vertexData[offset + 5] = blue;
      vertexData[offset + 6] = opacity;
      vertexData[offset + 7] = scaleX;
      vertexData[offset + 8] = scaleY;
      vertexData[offset + 9] = corner[0];
      vertexData[offset + 10] = corner[1];
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

    for (let i = 0; i < pointCloud.vertexCount; i += 1) {
      depths[i] = cameraDepth(pointCloud, i, view);
    }
    order.sort((a, b) => depths[b] - depths[a]);

    let dataOffset = 0;
    let renderedVertices = 0;
    for (const splatIndex of order) {
      if (depths[splatIndex] <= 0.02) continue;
      dataOffset = writeSplat(dataOffset, splatIndex);
      renderedVertices += SPLAT_CORNERS.length;
    }

    gl.viewport(0, 0, width, height);
    gl.clearColor(0.055, 0.045, 0.038, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      vertexData.subarray(0, dataOffset),
      gl.DYNAMIC_DRAW,
    );

    const stride = VERTEX_FLOATS * 4;
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(locations.color);
    gl.vertexAttribPointer(locations.color, 3, gl.FLOAT, false, stride, 3 * 4);
    gl.enableVertexAttribArray(locations.opacity);
    gl.vertexAttribPointer(locations.opacity, 1, gl.FLOAT, false, stride, 6 * 4);
    gl.enableVertexAttribArray(locations.scale);
    gl.vertexAttribPointer(locations.scale, 2, gl.FLOAT, false, stride, 7 * 4);
    gl.enableVertexAttribArray(locations.corner);
    gl.vertexAttribPointer(locations.corner, 2, gl.FLOAT, false, stride, 9 * 4);

    gl.uniform2f(locations.rotation, view.yaw, view.pitch);
    gl.uniform2f(locations.pan, view.panX, view.panY);
    gl.uniform1f(locations.distance, view.distance);
    gl.uniform1f(locations.aspect, width / height);

    gl.drawArrays(gl.TRIANGLES, 0, renderedVertices);
  }

  function dispose() {
    gl.deleteBuffer(vertexBuffer);
    gl.deleteProgram(program);
  }

  return { draw, dispose };
}

export default function PlyViewer({ src }) {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const dragRef = useRef(null);
  const [pointCloud, setPointCloud] = useState(null);
  const [view, setView] = useState(DEFAULT_VIEW);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function load() {
      setStatus("loading");
      setError("");
      setPointCloud(null);
      setView(DEFAULT_VIEW);

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
    if (!canvasRef.current || !pointCloud) return undefined;

    let renderer;
    try {
      renderer = createRenderer(canvasRef.current, pointCloud);
      rendererRef.current = renderer;
      renderer.draw(view);
    } catch (rendererError) {
      setStatus("failed");
      setError(rendererError.message);
      return undefined;
    }

    const resizeObserver = new ResizeObserver(() => renderer.draw(view));
    resizeObserver.observe(canvasRef.current);

    return () => {
      resizeObserver.disconnect();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [pointCloud]);

  useEffect(() => {
    rendererRef.current?.draw(view);
  }, [view]);

  function startDrag(event) {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      mode: event.shiftKey || event.altKey ? "pan" : "rotate",
      view,
    };
  }

  function moveDrag(event) {
    if (!dragRef.current) return;
    const drag = dragRef.current;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;

    if (drag.mode === "pan") {
      const panSpeed = 0.0022 * drag.view.distance;
      setView({
        ...drag.view,
        panX: drag.view.panX + dx * panSpeed,
        panY: drag.view.panY - dy * panSpeed,
      });
      return;
    }

    setView({
      ...drag.view,
      yaw: drag.view.yaw + dx * 0.008,
      pitch: clamp(drag.view.pitch + dy * 0.008, -1.45, 1.45),
    });
  }

  function endDrag(event) {
    if (dragRef.current) event.currentTarget.releasePointerCapture(event.pointerId);
    dragRef.current = null;
  }

  function zoom(event) {
    event.preventDefault();
    setView((current) => ({
      ...current,
      distance: clamp(current.distance * Math.exp(event.deltaY * 0.001), 0.5, 8),
    }));
  }

  function resetView() {
    setView(createInitialView(pointCloud));
  }

  return (
    <div className="viewer-card">
      <div
        className="viewer-canvas-wrap"
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onWheel={zoom}
        onDoubleClick={resetView}
      >
        <canvas ref={canvasRef} aria-label="3D Gaussian splat viewer" />
        {status !== "ready" && (
          <div className="viewer-overlay">
            {status === "loading" ? "Loading Gaussian splats..." : error}
          </div>
        )}
      </div>

      <div className="viewer-toolbar">
        <div>
          <p className="eyebrow">SPLAT VIEW</p>
          <p className="viewer-meta">
            {pointCloud
              ? `${formatCount(pointCloud.vertexCount)} Gaussian splats rendered`
              : "Preparing splat renderer"}
          </p>
        </div>
        <button type="button" className="ghost-button" onClick={resetView}>
          Reset view
        </button>
      </div>

      <p className="viewer-help">Drag to orbit. Scroll to zoom. Shift + drag to pan.</p>
    </div>
  );
}
