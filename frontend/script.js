const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

const askButton = document.getElementById("ask-button");
const questionInput = document.getElementById("question");
const strategyInput = document.getElementById("strategy");
const topKInput = document.getElementById("top-k");
const queryStatus = document.getElementById("query-status");

const resultSection = document.getElementById("result-section");

const answerElement = document.getElementById("answer");
const detailsButton = document.getElementById("details-button");
const detailsElement = document.getElementById("details");

const confidenceElement = document.getElementById("confidence");
const groundedElement = document.getElementById("grounded");
const latencyElement = document.getElementById("latency");
const strategyElement = document.getElementById("result-strategy");

const sourcesElement = document.getElementById("sources");
const chunksElement = document.getElementById("chunks");


uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        showStatus(uploadStatus, "Please select a file.", false);
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    showStatus(uploadStatus, "Uploading and indexing...", true);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        showStatus(
            uploadStatus,
            "Document uploaded successfully.",
            true
        );

    } catch (error) {
        showStatus(uploadStatus, error.message, false);
    }
});


askButton.addEventListener("click", async () => {

    const question = questionInput.value.trim();

    if (!question) {
        showStatus(queryStatus, "Please enter a question.", false);
        return;
    }

    const request = {
        question: question,
        strategy: strategyInput.value,
        top_k: Number(topKInput.value)
    };

    showStatus(
        queryStatus,
        "Searching documents and generating answer...",
        true
    );

    try {

        const response = await fetch("/api/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(request)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Query failed.");
        }

        displayResult(data);

        showStatus(
            queryStatus,
            "Answer generated successfully.",
            true
        );

    } catch (error) {
        showStatus(queryStatus, error.message, false);
    }
});


function displayResult(data) {

    resultSection.hidden = false;

    answerElement.textContent = data.response.answer;

    confidenceElement.textContent =
        `${(data.response.confidence * 100).toFixed(0)}%`;

    groundedElement.textContent =
        data.response.grounded ? "Yes" : "No";

    latencyElement.textContent =
        `${data.latency_s.toFixed(2)} seconds`;

    strategyElement.textContent =
        data.strategy;

    displaySources(data.response.sources);

    displayChunks(data.retrieved_chunks);

    // Hide technical details whenever a new answer is generated.
    detailsElement.hidden = true;
    detailsButton.textContent = "View Details";
}


detailsButton.addEventListener("click", () => {

    detailsElement.hidden = !detailsElement.hidden;

    detailsButton.textContent =
        detailsElement.hidden
            ? "View Details"
            : "Hide Details";
});


function displaySources(sources) {

    sourcesElement.innerHTML = "";

    if (!sources.length) {

        const item = document.createElement("li");
        item.textContent = "No verified sources.";
        sourcesElement.appendChild(item);

        return;
    }

    sources.forEach(source => {

        const item = document.createElement("li");

        const pageText =
            source.page === null
                ? ""
                : `, Page ${source.page}`;

        item.textContent =
            `${source.document}${pageText}, Chunk ${source.chunk_id}`;

        sourcesElement.appendChild(item);
    });
}


function displayChunks(chunks) {

    chunksElement.innerHTML = "";

    chunks.forEach(chunk => {

        const div = document.createElement("div");

        div.className = "chunk";

        div.innerHTML = `
            <strong>${escapeHtml(chunk.chunk_id)}</strong>
            <br>
            Score: ${chunk.score.toFixed(4)}
            <br>
            ${escapeHtml(chunk.text)}
        `;

        chunksElement.appendChild(div);
    });
}


function showStatus(element, message, success) {

    element.textContent = message;

    element.className =
        success
            ? "status-success"
            : "status-error";
}


function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}