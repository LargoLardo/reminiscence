import React, { useEffect, useState } from "react";
import PlyViewer from "./PlyViewer";

const terminalStatuses = new Set(["complete", "failed"]);
const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path) {
  return `${apiBaseUrl}${path}`;
}

function MomentComparison({ previewSrc, sourceSrc }) {
  const [previewAvailable, setPreviewAvailable] = useState(Boolean(previewSrc));
  const [sourceAvailable, setSourceAvailable] = useState(Boolean(sourceSrc));

  useEffect(() => {
    setPreviewAvailable(Boolean(previewSrc));
    setSourceAvailable(Boolean(sourceSrc));
  }, [previewSrc, sourceSrc]);

  if (!previewAvailable && !sourceAvailable) return null;

  return (
    <div className="comparison-grid">
      {previewAvailable && (
        <article className="media-card">
          <p className="eyebrow">MODEL RENDER</p>
          <img
            src={previewSrc}
            alt="Rendered preview from the reconstructed Gaussian splat"
            onError={() => setPreviewAvailable(false)}
          />
        </article>
      )}
      {sourceAvailable && (
        <article className="media-card">
          <p className="eyebrow">ORIGINAL CLIP</p>
          <video
            src={sourceSrc}
            controls
            muted
            loop
            playsInline
            onError={() => setSourceAvailable(false)}
          />
        </article>
      )}
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(() => {
    const existingJobId = new URLSearchParams(window.location.search).get("job");
    return existingJobId ? { id: existingJobId, status: "processing" } : null;
  });
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (!job?.id || terminalStatuses.has(job.status)) return undefined;

    let cancelled = false;

    async function poll() {
      try {
        const response = await fetch(apiUrl(`/api/v1/moments/${job.id}`));
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Could not read job status");
        if (!cancelled) setJob(body);
      } catch (pollError) {
        if (!cancelled) setError(pollError.message);
      }
    }

    poll();
    const timer = setInterval(poll, 3000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.id, job?.status]);

  async function upload(event) {
    event.preventDefault();
    if (!file) return;

    setError("");
    setJob(null);
    setUploading(true);
    window.history.replaceState(null, "", window.location.pathname);

    const form = new FormData();
    form.append("video", file);
    form.append("captured_at", new Date().toISOString());
    form.append("duration", "0");

    try {
      const response = await fetch(apiUrl("/api/v1/moments"), {
        method: "POST",
        body: form,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Upload failed");
      setJob(body);
      window.history.replaceState(null, "", `?job=${encodeURIComponent(body.id)}`);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setUploading(false);
    }
  }

  async function download() {
    setError("");
    try {
      const response = await fetch(apiUrl(`/api/v1/moments/${job.id}/splat`));
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || "Download failed");
      }
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${job.dataset_name}.ply`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (downloadError) {
      setError(downloadError.message);
    }
  }

  const splatUrl = job?.id ? apiUrl(`/api/v1/moments/${job.id}/splat`) : "";
  const sourceUrl = job?.id ? apiUrl(`/api/v1/moments/${job.id}/source`) : "";
  const renderPreviewUrl = job?.render_preview_url
    ? apiUrl(`/api/v1/moments/${job.id}/preview`)
    : "";

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">REMINISCENCE</p>
        <h1>Turn a video into a place you can revisit.</h1>
        <p className="lede">
          Upload a slow walk around a memory. We will reconstruct it into a
          Gaussian splat ready for Unity.
        </p>
      </section>

      <section className="panel">
        <form onSubmit={upload}>
          <label className="file-picker">
            <span>{file ? file.name : "Choose a video"}</span>
            <input
              type="file"
              accept="video/*"
              onChange={(event) => setFile(event.target.files[0])}
            />
          </label>
          <button disabled={!file || uploading} type="submit">
            {uploading ? "Uploading..." : "Create memory"}
          </button>
        </form>

        {job && (
          <div className="job">
            <div>
              <p className="eyebrow">RECONSTRUCTION</p>
              <h2>{job.status}</h2>
            </div>
            <span className={`status ${job.status}`} />
            {job.registered_image_count != null && (
              <p>{job.registered_image_count} camera views registered</p>
            )}
            {job.error && <p className="error">{job.error}</p>}
            {job.status === "complete" && (
              <button type="button" onClick={download}>Download .ply for Unity</button>
            )}
          </div>
        )}

        {error && <p className="error">{error}</p>}
      </section>

      {job?.status === "complete" && (
        <section className="viewer-panel">
          <div className="viewer-heading">
            <p className="eyebrow">MEMORY PREVIEW</p>
            <h2>Compare it to the source</h2>
          </div>
          <MomentComparison previewSrc={renderPreviewUrl} sourceSrc={sourceUrl} />
          <div className="viewer-heading compact">
            <p className="eyebrow">INTERACTIVE VIEW</p>
            <h2>Explore the Gaussian splat</h2>
          </div>
          <PlyViewer src={splatUrl} />
        </section>
      )}
    </main>
  );
}
