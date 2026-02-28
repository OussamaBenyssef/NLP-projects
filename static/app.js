/* ═══════════════════════════════════════════════════════
   app.js — Chat Multilingue Frontend
   ═══════════════════════════════════════════════════════ */

(function () {
    "use strict";

    // ── DOM References ──
    const form = document.getElementById("chat-form");
    const input = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const messagesEl = document.getElementById("messages");
    const detailsPanel = document.getElementById("details-panel");
    const detailsContent = document.getElementById("details-content");
    const toggleBtn = document.getElementById("toggle-details");

    // ── State ──
    let isWaiting = false;
    let welcomeVisible = true;

    // ── RTL Labels ──
    const RTL_LABELS = new Set(["AR_DAR", "AR_MSA"]);

    // ── Label display names ──
    const LABEL_NAMES = {
        AR_DAR: "Darija",
        AR_MSA: "Arabe MSA",
        EN: "Anglais",
        FR: "Français",
    };

    // ═══════════════════════════════════════════
    // Toggle Details Panel + localStorage
    // ═══════════════════════════════════════════

    function initToggle() {
        const saved = localStorage.getItem("details-visible");
        // Default: visible on desktop, hidden on mobile
        const isMobile = window.innerWidth <= 768;
        const visible = saved !== null ? saved === "true" : !isMobile;
        setDetailsVisible(visible);

        toggleBtn.addEventListener("click", function () {
            const isHidden = detailsPanel.classList.contains("hidden");
            setDetailsVisible(isHidden);
        });
    }

    function setDetailsVisible(visible) {
        if (visible) {
            detailsPanel.classList.remove("hidden");
            toggleBtn.classList.add("active");
        } else {
            detailsPanel.classList.add("hidden");
            toggleBtn.classList.remove("active");
        }
        localStorage.setItem("details-visible", String(visible));
    }

    // ═══════════════════════════════════════════
    // HTML Escaping
    // ═══════════════════════════════════════════

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // ═══════════════════════════════════════════
    // Auto-resize textarea
    // ═══════════════════════════════════════════

    function autoResize() {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
    }

    // ═══════════════════════════════════════════
    // Scroll to bottom
    // ═══════════════════════════════════════════

    function scrollToBottom() {
        requestAnimationFrame(function () {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    }

    // ═══════════════════════════════════════════
    // Remove welcome message
    // ═══════════════════════════════════════════

    function removeWelcome() {
        if (welcomeVisible) {
            const w = messagesEl.querySelector(".welcome-message");
            if (w) w.remove();
            welcomeVisible = false;
        }
    }

    // ═══════════════════════════════════════════
    // Add message to chat
    // ═══════════════════════════════════════════

    function addMessage(role, text, label) {
        removeWelcome();

        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + role;

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        // RTL for Arabic labels (bot only)
        if (role === "bot" && label && RTL_LABELS.has(label)) {
            bubble.setAttribute("dir", "rtl");
        }

        // Label badge for bot
        if (role === "bot" && label) {
            const badge = document.createElement("div");
            badge.className = "message-label label-" + label;
            badge.textContent = LABEL_NAMES[label] || label;
            bubble.appendChild(badge);
        }

        // Text content
        const textNode = document.createElement("div");
        textNode.innerHTML = escapeHtml(text);
        bubble.appendChild(textNode);

        msgDiv.appendChild(bubble);
        messagesEl.appendChild(msgDiv);
        scrollToBottom();

        return msgDiv;
    }

    function addErrorMessage(text) {
        removeWelcome();

        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble error-bubble";
        bubble.textContent = "Erreur : " + text;

        msgDiv.appendChild(bubble);
        messagesEl.appendChild(msgDiv);
        scrollToBottom();
    }

    // ═══════════════════════════════════════════
    // Loading indicator
    // ═══════════════════════════════════════════

    function showLoading() {
        removeWelcome();

        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot";
        msgDiv.id = "loading-msg";

        const dots = document.createElement("div");
        dots.className = "loading-dots";
        dots.innerHTML = "<span></span><span></span><span></span>";

        msgDiv.appendChild(dots);
        messagesEl.appendChild(msgDiv);
        scrollToBottom();
    }

    function hideLoading() {
        const el = document.getElementById("loading-msg");
        if (el) el.remove();
    }

    // ═══════════════════════════════════════════
    // Render Details Panel
    // ═══════════════════════════════════════════

    function renderDetails(data) {
        let html = "";

        // ── 1. Language detection card ──
        const label = data.label || "—";
        const confidence = data.confidence || 0;
        const confPct = (confidence * 100).toFixed(1);
        const script = data.script;
        const labelName = LABEL_NAMES[label] || label;

        html += '<div class="detail-card">';
        html += '<div class="detail-card-title">Langue d\u00e9tect\u00e9e</div>';
        html += '<div class="detail-value">' + escapeHtml(labelName) + '</div>';
        html += '<div class="detail-sub">';
        html += '<span class="message-label label-' + label + '">' + label + "</span>";
        html += " &nbsp; Confiance : " + confPct + "%";
        if (script) {
            html += " &nbsp;│&nbsp; Script : " + escapeHtml(script);
        }
        html += "</div></div>";

        // ── 2. Scores card ──
        if (data.scores) {
            html += '<div class="detail-card">';
            html += '<div class="detail-card-title">Probabilités par classe</div>';

            // Sort by score descending
            const entries = Object.entries(data.scores).sort(function (a, b) {
                return b[1] - a[1];
            });

            for (var i = 0; i < entries.length; i++) {
                var scoreLbl = entries[i][0];
                var scoreVal = entries[i][1];
                var pct = (scoreVal * 100).toFixed(1);
                var barWidth = Math.max(scoreVal * 100, 1);

                html += '<div class="score-row">';
                html += '<span class="score-label">' + scoreLbl + "</span>";
                html += '<div class="score-bar-bg">';
                html +=
                    '<div class="score-bar-fill bar-' +
                    scoreLbl +
                    '" style="width:' +
                    barWidth +
                    '%"></div>';
                html += "</div>";
                html += '<span class="score-value">' + pct + "%</span>";
                html += "</div>";
            }

            html += "</div>";
        }

        // ── 3. Timings card ──
        if (data.timings) {
            html += '<div class="detail-card">';
            html += '<div class="detail-card-title">Temps d\'inférence</div>';

            html += '<div class="timing-row">';
            html += '<span class="timing-label">Détection</span>';
            html +=
                '<span class="timing-value">' +
                (data.timings.detect_ms || 0).toFixed(0) +
                " ms</span>";
            html += "</div>";

            html += '<div class="timing-row">';
            html += '<span class="timing-label">Génération</span>';
            html +=
                '<span class="timing-value">' +
                formatTime(data.timings.generate_ms) +
                "</span>";
            html += "</div>";

            var totalMs =
                (data.timings.detect_ms || 0) + (data.timings.generate_ms || 0);
            html += '<div class="timing-row">';
            html += '<span class="timing-label">Total</span>';
            html +=
                '<span class="timing-value">' + formatTime(totalMs) + "</span>";
            html += "</div>";

            html += "</div>";
        }

        // ── 4. Model info card ──
        if (data.model) {
            html += '<div class="detail-card">';
            html += '<div class="detail-card-title">Modèles utilisés</div>';
            html += '<div class="model-info">';
            html +=
                "Détecteur : <code>" +
                escapeHtml(data.model.detector || "—") +
                "</code><br>";
            html +=
                "Générateur : <code>" +
                escapeHtml(data.model.generator || "—") +
                "</code>";
            if (data.prompt_key) {
                html +=
                    "<br>Prompt : <code>" +
                    escapeHtml(data.prompt_key) +
                    "</code>";
            }
            html += "</div></div>";
        }

        detailsContent.innerHTML = html;
    }

    function formatTime(ms) {
        if (!ms) return "—";
        if (ms >= 1000) {
            return (ms / 1000).toFixed(1) + " s";
        }
        return ms.toFixed(0) + " ms";
    }

    // ═══════════════════════════════════════════
    // Send Message
    // ═══════════════════════════════════════════

    async function sendMessage(text) {
        if (isWaiting || !text.trim()) return;

        isWaiting = true;
        sendBtn.disabled = true;

        // Add user message
        addMessage("user", text);
        input.value = "";
        autoResize();

        // Show loading
        showLoading();

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text }),
            });

            hideLoading();

            if (!resp.ok) {
                var errData;
                try {
                    errData = await resp.json();
                } catch (_) {
                    errData = {};
                }
                addErrorMessage(
                    errData.error || "Erreur serveur (" + resp.status + ")"
                );
                return;
            }

            const data = await resp.json();

            if (data.error) {
                addErrorMessage(data.error);
                return;
            }

            // Add bot message
            addMessage("bot", data.answer, data.label);

            // Render details panel
            renderDetails(data);
        } catch (err) {
            hideLoading();
            addErrorMessage(
                "Erreur de connexion. Vérifiez que le serveur est lancé."
            );
            console.error("Chat error:", err);
        } finally {
            isWaiting = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }

    // ═══════════════════════════════════════════
    // Event Listeners
    // ═══════════════════════════════════════════

    form.addEventListener("submit", function (e) {
        e.preventDefault();
        sendMessage(input.value);
    });

    // Enter to send, Shift+Enter for newline
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage(input.value);
        }
    });

    input.addEventListener("input", autoResize);

    // ── Init ──
    initToggle();
    input.focus();
})();
