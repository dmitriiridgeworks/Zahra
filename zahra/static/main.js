let mode = "int";

function setMode(newMode) {
    if (mode === newMode) return;
    mode = newMode;
    const body = document.body;
    if (mode === "int") body.classList.add("int-mode"), body.classList.remove("ext-mode");
    else body.classList.add("ext-mode"), body.classList.remove("int-mode");
}

const questionInput = document.getElementById("question");
const answerDiv = document.getElementById("answer");

// FIXED START HEIGHT (one reasonable line)
const initialHeight = 50; // pixels
questionInput.style.height = initialHeight + 'px';

// AUTO-RESIZE TO FIT CONTENT
function adjustAnswerHeight() {
    questionInput.style.height = 'auto'; // reset to measure content
    const newHeight = Math.max(questionInput.scrollHeight, initialHeight);
    questionInput.style.height = newHeight + 'px';

    const container = document.querySelector('.container');
    const containerHeight = container.clientHeight;
    const questionHeight = questionInput.offsetHeight + 12; // margin
    answerDiv.style.height = (containerHeight - questionHeight - 120) + 'px';
}

// Expand/shrink as user types
questionInput.addEventListener('input', adjustAnswerHeight);

// SEND ON ENTER
questionInput.addEventListener("keypress", function(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});

async function ask() {
    const question = questionInput.value.trim();
    if (!question) return;

    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.textContent = question;
    answerDiv.appendChild(userBubble);

    const intentBubble = document.createElement("div");
    intentBubble.className = "chat-bubble ai intent";
    answerDiv.appendChild(intentBubble);

    let answerBubble = null;
    let phase = "intent";

    answerDiv.scrollTop = answerDiv.scrollHeight;
    questionInput.value = '';
    questionInput.style.height = initialHeight + 'px'; // reset to initial height

    try {
        const response = await fetch("/ask_stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode, question }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            const lines = text.split("\n\n").filter((l) => l.startsWith("data:"));

            for (const line of lines) {
                const content = line.replace("data: ", "");

                if (content.includes("---")) {
                    phase = "answer";
                    answerBubble = document.createElement("div");
                    answerBubble.className = "chat-bubble ai answer";
                    answerDiv.appendChild(answerBubble);
                    continue;
                }

                if (phase === "intent") intentBubble.textContent += content;
                else if (phase === "answer") answerBubble.textContent += content;

                answerDiv.scrollTop = answerDiv.scrollHeight;
            }
        }
    } catch (err) {
        console.error(err);
    }
}

// INITIAL adjustment
window.addEventListener('load', adjustAnswerHeight);
