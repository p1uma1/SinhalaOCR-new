import { useState } from "react";
import { ApiError, ocrCharacter, ocrDocument, ocrLine } from "../api";
import type {
  CharacterOCRResult,
  DocumentOCRResult,
  HealthStatus,
  TabId,
} from "../types";
import { UploadZone } from "./UploadZone";

interface OCRStudioProps {
  health: HealthStatus | null;
  showToast: (message: string, type?: "success" | "error" | "info") => void;
}

const TABS: { id: TabId; label: string }[] = [
  { id: "line", label: "Line OCR" },
  { id: "document", label: "Document OCR" },
  { id: "character", label: "Character" },
];

export function OCRStudio({ health, showToast }: OCRStudioProps) {
  const [activeTab, setActiveTab] = useState<TabId>("line");
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("Processing…");

  const [lineFile, setLineFile] = useState<File | null>(null);
  const [lineText, setLineText] = useState("");

  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [documentResult, setDocumentResult] = useState<DocumentOCRResult | null>(null);

  const [characterFile, setCharacterFile] = useState<File | null>(null);
  const [characterResult, setCharacterResult] = useState<CharacterOCRResult | null>(null);

  const run = async (action: () => Promise<void>, message: string) => {
    setLoading(true);
    setLoadingMessage(message);
    try {
      await action();
    } catch (error) {
      const msg =
        error instanceof ApiError
          ? error.message
          : "Unexpected error. Please try again.";
      showToast(msg, "error");
    } finally {
      setLoading(false);
    }
  };

  const copyText = async (text: string) => {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard", "success");
  };

  const stage2Missing = health && !health.stage2_ready;
  const stage1Missing = health && !health.stage1_ready;

  return (
    <section className="section studio-section" id="studio">
      <div className="container">
        <div className="section-head">
          <p className="eyebrow">OCR Studio</p>
          <h2>Try the models live</h2>
          <p>Upload an image and get Sinhala text in seconds.</p>
        </div>

        <div className="studio-shell">
          <div className="tabs" role="tablist" aria-label="OCR modes">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={`tab ${activeTab === tab.id ? "active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === "line" && (
            <div className="studio-panel" role="tabpanel">
              {stage2Missing && (
                <div className="alert warn">
                  Stage 2 model is not trained yet. Run <code>train_stage2.py</code> first.
                </div>
              )}
              <div className="studio-grid">
                <div className="card">
                  <h3>Line image input</h3>
                  <p className="card-hint">One horizontal line of Sinhala text per image.</p>
                  <UploadZone
                    id="line-upload"
                    label="Drag & drop or click to browse"
                    hint="Best results with cropped line images like SinOCR training data"
                    icon="↑"
                    onFileSelect={setLineFile}
                    onError={(m) => showToast(m, "error")}
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    disabled={!lineFile || loading}
                    onClick={() =>
                      run(async () => {
                        if (!lineFile) return;
                        const result = await ocrLine(lineFile);
                        setLineText(result.text);
                        showToast("Line recognized", "success");
                      }, "Recognizing line…")
                    }
                  >
                    Recognize line
                  </button>
                </div>
                <div className="card">
                  <div className="card-head">
                    <h3>Recognized text</h3>
                    {lineText && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => copyText(lineText)}
                      >
                        Copy
                      </button>
                    )}
                  </div>
                  <div className="result-box sinhala">
                    {lineText || (
                      <p className="placeholder">
                        Upload a line image and run recognition to see Sinhala output here.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "document" && (
            <div className="studio-panel" role="tabpanel">
              {stage2Missing && (
                <div className="alert warn">
                  Stage 2 model is not trained yet. Run <code>train_stage2.py</code> first.
                </div>
              )}
              <div className="studio-grid">
                <div className="card">
                  <h3>Document page input</h3>
                  <p className="card-hint">
                    Full page — binary mask finds line &amp; word boxes; OCR runs left-to-right
                    with shared decoding context.
                  </p>
                  <UploadZone
                    id="document-upload"
                    label="Drag & drop a document page"
                    hint="Works with scanned pages, photos, and handwritten sheets"
                    icon="▤"
                    onFileSelect={setDocumentFile}
                    onError={(m) => showToast(m, "error")}
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    disabled={!documentFile || loading}
                    onClick={() =>
                      run(async () => {
                        if (!documentFile) return;
                        const result = await ocrDocument(documentFile);
                        setDocumentResult(result);
                        showToast(`Processed ${result.line_count} lines`, "success");
                      }, "Processing document…")
                    }
                  >
                    Process document
                  </button>
                </div>
                <div className="card">
                  <div className="card-head">
                    <h3>Full transcription</h3>
                    {documentResult?.full_text && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => copyText(documentResult.full_text)}
                      >
                        Copy all
                      </button>
                    )}
                  </div>
                  <div className="result-box sinhala">
                    {documentResult?.full_text || (
                      <p className="placeholder">
                        Full document text will appear here after processing.
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {documentResult && documentResult.lines.length > 0 && (
                <div className="card lines-panel">
                  <h3>
                    Line-by-line results{" "}
                    <span className="badge">{documentResult.line_count} lines</span>
                    {documentResult.word_count != null && documentResult.word_count > 0 && (
                      <span className="badge badge-muted">
                        {documentResult.word_count} words
                      </span>
                    )}
                  </h3>
                  <div className="lines-list">
                    {documentResult.lines.map((line) => (
                      <article key={line.index} className="line-row">
                        <img
                          src={`data:image/png;base64,${line.image_base64}`}
                          alt={`Line ${line.index + 1}`}
                        />
                        <div className="line-body">
                          <span className="line-label">Line {line.index + 1}</span>
                          <p className="sinhala line-text">{line.text || "—"}</p>
                          {line.words && line.words.length > 0 && (
                            <ul className="word-list">
                              {line.words.map((word) => (
                                <li key={word.index} className="word-chip">
                                  <img
                                    src={`data:image/png;base64,${word.image_base64}`}
                                    alt={`Word ${word.index + 1}`}
                                    className="word-thumb"
                                  />
                                  <span className="sinhala word-text">
                                    {word.text || "—"}
                                  </span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              )}

              {documentResult?.debug?.original && (
                <div className="debug-grid">
                  <div className="card">
                    <h3>Original</h3>
                    <img
                      className="debug-image"
                      src={`data:image/png;base64,${documentResult.debug.original}`}
                      alt="Original document"
                    />
                  </div>
                  <div className="card">
                    <h3>Binary mask &amp; boxes</h3>
                    <img
                      className="debug-image"
                      src={`data:image/png;base64,${documentResult.debug.boxes || documentResult.debug.deskewed}`}
                      alt="Segmentation boxes"
                    />
                    <p className="card-hint">Green = lines, red = word boxes from binary mask</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "character" && (
            <div className="studio-panel" role="tabpanel">
              {stage1Missing && (
                <div className="alert warn">
                  Stage 1 model is not trained yet. Run <code>train_stage1.py</code> first.
                </div>
              )}
              <div className="studio-grid">
                <div className="card">
                  <h3>Character crop input</h3>
                  <p className="card-hint">Single Sinhala character — Stage 1 classifier demo.</p>
                  <UploadZone
                    id="character-upload"
                    label="Drag & drop one character"
                    hint="454-class ViT classifier trained on Dataset454"
                    icon="අ"
                    onFileSelect={setCharacterFile}
                    onError={(m) => showToast(m, "error")}
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    disabled={!characterFile || loading}
                    onClick={() =>
                      run(async () => {
                        if (!characterFile) return;
                        const result = await ocrCharacter(characterFile);
                        setCharacterResult(result);
                        showToast("Character classified", "success");
                      }, "Classifying character…")
                    }
                  >
                    Classify character
                  </button>
                </div>
                <div className="card">
                  <h3>Top predictions</h3>
                  <div className="result-box">
                    {!characterResult ? (
                      <p className="placeholder">
                        Top-5 character predictions with confidence scores will appear here.
                      </p>
                    ) : (
                      <ul className="prediction-list">
                        {characterResult.predictions.map((item, index) => (
                          <li
                            key={`${item.label}-${index}`}
                            className={`prediction-item ${index === 0 ? "top" : ""}`}
                          >
                            <span className="sinhala prediction-label">
                              {item.character ?? item.label}
                            </span>
                            <span className="prediction-score">{item.confidence}%</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {loading && (
        <div className="overlay" role="status" aria-live="polite">
          <div className="spinner" />
          <p>{loadingMessage}</p>
        </div>
      )}
    </section>
  );
}
