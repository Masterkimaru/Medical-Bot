// Emergency keywords matching backend
const EMERGENCY_KEYWORDS = [
  'faint', 'seizure', 'choking', 'bleeding', 
  'nosebleed', 'unconscious', 'heart attack', 'overdose', 'stroke'
];

// First aid protocols for emergency responses
const FIRST_AID_PROTOCOLS = {
  'choking': `
    <div class="emergency-alert">
      <div class="emergency-header">🚨 CHOKING EMERGENCY</div>
      <strong>Critical Actions:</strong>
      <div class="bullet-point">Ask "Can you speak?"</div>
      <div class="bullet-point">Give 5 back blows between shoulder blades</div>
      <div class="bullet-point">Perform 5 abdominal thrusts (Heimlich maneuver)</div>
      <div class="bullet-point">Call emergency services if person becomes unconscious</div>
    </div>
  `,
  'heart attack': `
    <div class="emergency-alert">
      <div class="emergency-header">🚨 HEART ATTACK EMERGENCY</div>
      <strong>Critical Actions:</strong>
      <div class="bullet-point">Call emergency services immediately</div>
      <div class="bullet-point">Have person sit down and stay calm</div>
      <div class="bullet-point">Give aspirin if available and not allergic</div>
      <div class="bullet-point">Prepare to perform CPR if person becomes unresponsive</div>
    </div>
  `,
  'default': `
    <div class="emergency-alert">
      <div class="emergency-header">🚨 EMERGENCY SITUATION</div>
      <div class="bullet-point">Call emergency services immediately</div>
      <div class="bullet-point">Stay with the person and monitor their condition</div>
      <div class="bullet-point">Do not move the person unless absolutely necessary</div>
    </div>
  `
};

function detectEmergency(query) {
  const lowerQuery = query.toLowerCase();
  for (const keyword of EMERGENCY_KEYWORDS) {
    if (lowerQuery.includes(keyword)) {
      return keyword;
    }
  }
  return null;
}

function getEmergencyProtocol(keyword) {
  return FIRST_AID_PROTOCOLS[keyword] || FIRST_AID_PROTOCOLS['default'];
}

/**
 * Transform a YouTube URL (e.g., "youtu.be" or "watch?v=") into the embed format.
 */
function transformYoutubeUrl(url) {
  if (url.includes("youtu.be/")) {
    // For shortened URLs, extract the video ID and form the embed URL.
    const parts = url.split("youtu.be/");
    const idAndQuery = parts[1]; // Might include query parameters, e.g., "?si=..."
    return "https://www.youtube.com/embed/" + idAndQuery;
  } else if (url.includes("watch?v=")) {
    // Replace watch?v= with embed/
    return url.replace("watch?v=", "embed/");
  }
  // If the URL is already in embed format or another type, return as-is.
  return url;
}

function sendMessage() {
  const input = document.getElementById('user-input');
  const query = input.value.trim();
  if (!query) return;

  // Capture the last four chat entries for conversation history
  const chatHistory = Array.from(document.querySelectorAll('.user-message, .bot-message'))
    .slice(-4)
    .map(msg => {
      const isUser = msg.classList.contains('user-message');
      return `${isUser ? 'Patient' : 'Doctor'}: ${msg.querySelector('.message').textContent}`;
    });

  addUserMessage(query);
  input.value = '';

  const loadingMsg = addBotMessage("Analyzing your symptoms...");

  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ query, history: chatHistory })
  })
    .then(res => res.json())
    .then(data => {
      loadingMsg.remove();

      // Format the medical response coming from the backend
      let response = formatMedicalResponse(data.response);

      // Append emergency protocol if the response category is emergency
      if (data.category === 'emergency') {
        const emergencyType = detectEmergency(query);
        response += getEmergencyProtocol(emergencyType);
      }
      
      // Append the video embed if a video link is returned from backend.
      if (data.video_link) {
        // Transform the URL into the embed format.
        const embedUrl = transformYoutubeUrl(data.video_link);
        response += `
          <div class="video-container">
            <iframe src="${embedUrl}" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
            </iframe>
          </div>
        `;
      }
      
      addBotMessage(response);
    })
    .catch(err => {
      loadingMsg.remove();
      addBotMessage("Error processing your request. Please try again.");
      console.error(err);
    });
}

function formatMedicalResponse(text) {
  // Format section headers
  text = text
    .replace(/\[Emergency Steps\]/g, '<div class="section-title">🚨 EMERGENCY STEPS</div>')
    .replace(/\[Possible Emergency Causes\]/g, '<div class="section-title">POSSIBLE CAUSES</div>')
    .replace(/\[When to Call EMS\]/g, '<div class="warning-box section-title">⚠️ WHEN TO CALL EMERGENCY SERVICES</div>')
    .replace(/\[Do Not\]/g, '<div class="section-title">🚫 DO NOT</div>')
    .replace(/\[Follow-up Questions\]/g, '<div class="section-title">FOLLOW-UP QUESTIONS</div>')
    .replace(/\[Possible Conditions\]/g, '<div class="section-title">POSSIBLE CONDITIONS</div>')
    .replace(/\[Recommended Actions\]/g, '<div class="section-title">RECOMMENDED ACTIONS</div>')
    .replace(/\[Medication & Treatment\]/g, '<div class="section-title">MEDICATION & TREATMENT</div>')
    .replace(/\[PHQ-2 Screening\]/g, '<div class="section-title">DEPRESSION SCREENING</div>')
    .replace(/\[Coping Strategies\]/g, '<div class="section-title">COPING STRATEGIES</div>');

  // Format bullet points and line breaks
  text = text
    .replace(/\n/g, '<br>')
    .replace(/- (.*?)(<br>|$)/g, '<div class="bullet-point">$1</div>')
    .replace(/⚠️ (.*?)(<br>|$)/g, '<div class="bullet-point warning">⚠️ $1</div>')
    .replace(/🚑 (.*?)(<br>|$)/g, '<div class="bullet-point emergency">🚑 $1</div>');

  return text;
}

// ================== Image Upload ==================
document.getElementById('image-btn').addEventListener('click', () =>
  document.getElementById('image-upload').click()
);
document.getElementById('image-upload').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) {
    addBotMessage("Please upload an image smaller than 5MB.");
    return;
  }
  const reader = new FileReader();
  reader.onload = async () => {
    const base64 = reader.result.split(',')[1];
    addUserMessage("Uploaded medical image for analysis");
    const loading = addBotMessage("Analyzing medical image...");
    try {
      const res = await fetch('/analyze-image', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ image: base64 })
      });
      const data = await res.json();
      loading.remove();
      addBotMessage(data.response);
    } catch {
      loading.remove();
      addBotMessage("Image analysis failed. Please try again.");
    }
  };
  reader.readAsDataURL(file);
});

// ================== BMI Calculator ==================
function calculateBmi() {
  const w = parseFloat(document.getElementById('weight').value);
  const h = parseFloat(document.getElementById('height').value);
  if (!w || !h) { alert("Please enter both weight and height"); return; }
  fetch('/calculate-bmi', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ weight: w, height: h })
  })
  .then(res => res.json())
  .then(data => {
    const div = document.getElementById('bmi-result');
    div.innerHTML = `
      <h3>Your BMI: ${data.bmi}</h3>
      <p>Category: ${data.category}</p>
      <p>${data.interpretation}</p>
      <div class="bmi-scale">
        <div class="scale-item ${data.bmi < 18.5 ? 'active' : ''}">Underweight (<18.5)</div>
        <div class="scale-item ${(18.5 <= data.bmi && data.bmi < 25) ? 'active' : ''}">Normal (18.5-24.9)</div>
        <div class="scale-item ${(25 <= data.bmi && data.bmi < 30) ? 'active' : ''}">Overweight (25-29.9)</div>
        <div class="scale-item ${data.bmi >= 30 ? 'active' : ''}">Obese (30+)</div>
      </div>`;
    addUserMessage(`BMI calculation: Weight ${w}kg, Height ${h}cm`);
    addBotMessage(`Your BMI is ${data.bmi} (${data.category}). ${data.interpretation}`);
  })
  .catch(() => alert("Error calculating BMI. Please try again."));
}

function showBmiCalculator() {
  document.getElementById('bmi-modal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}

function closeBmiModal() {
  document.getElementById('bmi-modal').style.display = 'none';
  document.body.style.overflow = 'auto';
}

// ================== Core Chat Functions ==================
function addUserMessage(text) {
  const historyContainer = document.getElementById('chat-history');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'user-message';
  messageDiv.innerHTML = `<div class="message">${text}</div>`;
  historyContainer.appendChild(messageDiv);
  historyContainer.scrollTop = historyContainer.scrollHeight;
}

function addBotMessage(html) {
  const historyContainer = document.getElementById('chat-history');
  const messageWrapper = document.createElement('div');
  messageWrapper.className = 'bot-message';
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message';
  messageDiv.innerHTML = html;
  messageWrapper.appendChild(messageDiv);
  
  const btn = document.createElement('button');
  btn.className = 'icon-btn speech-response-btn';
  btn.setAttribute('data-speaking', 'false');
  btn.title = 'Speak Response';
  btn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" 
         viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
      <line x1="12" y1="19" x2="12" y2="23"></line>
      <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>`;
  btn.addEventListener('click', () => toggleResponseSpeech(btn, messageDiv.textContent));
  
  messageWrapper.appendChild(btn);
  historyContainer.appendChild(messageWrapper);
  historyContainer.scrollTop = historyContainer.scrollHeight;
  return messageWrapper;
}

// ================== Mood Tracker & CBT Functions ==================
document.getElementById('mood-rating').addEventListener('input', function() {
  document.getElementById('mood-value').textContent = this.value;
});

function openMoodTracker() {
  document.getElementById('mood-modal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}

function closeMoodTracker() {
  document.getElementById('mood-modal').style.display = 'none';
  document.body.style.overflow = 'auto';
}

function submitMoodRating() {
  const rating = parseInt(document.getElementById('mood-rating').value);
  const comment = document.getElementById('mood-comment').value.trim();
  
  let response = `Mood rating: ${rating}/10`;
  if(comment) response += ` — ${comment}`;
  
  addUserMessage(response);
  
  let botResponse = "Thanks for sharing how you're feeling. ";
  if(rating <= 4) {
    botResponse += `
      <div class="mental-health-alert">
        <h4>💚 You Matter</h4>
        <p>1. Try the 4-7-8 breathing: Inhale 4s, hold 7s, exhale 8s</p>
        <p>2. Contact a trusted friend/family member right now</p>
        <p>3. Crisis hotline: <button onclick="toggleHotline()">Show Numbers</button>
          <div id="hotline-numbers" style="display:none">
            US: 988 • UK: 116123 • IN: 9152987821
          </div>
        </p>
        <p>Would you like help finding a <u>therapist near you</u>?</p>
        <button class="therapy-search-btn" onclick="searchTherapists()">Yes, Find Help</button>
      </div>`;
  }
  else if(rating <= 6) {
    botResponse += `Consider trying a <button onclick="openCbtExercise()">CBT Exercise</button> to process your feelings.`;
  }
  else {
    botResponse += "Glad to hear you're doing well! Keep tracking your mood daily.";
  }
  
  addBotMessage(botResponse);
  closeMoodTracker();
}

function toggleHotline() {
  const el = document.getElementById('hotline-numbers');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function searchTherapists() {
  addBotMessage("I can help you find mental health professionals. What's your location (city/zip)?");
}

function openCbtExercise() {
  document.getElementById('cbt-modal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}

function closeCbtExercise() {
  document.getElementById('cbt-modal').style.display = 'none';
  document.body.style.overflow = 'auto';
}

function submitCbtExercise() {
  const s = document.getElementById('situation').value.trim();
  const t = document.getElementById('thoughts').value.trim();
  const e = document.getElementById('emotions').value.trim();
  const a = document.getElementById('alternative').value.trim();
  addUserMessage(`CBT Exercise:\nSituation: ${s}\nThoughts: ${t}\nEmotions: ${e}\nAlternative: ${a}`);
  
  let response = "Thanks for completing the CBT exercise. ";
  if(e.toLowerCase().includes('depress') || e.toLowerCase().includes('sad') || parseInt(e.split('/')[0]) > 6) {
    response += `
      <div class="mental-health-alert">
        <p>I notice you're experiencing strong emotions. Remember:</p>
        <p>• Feelings are temporary, even when they feel overwhelming</p>
        <p>• Consider scheduling a check-in with a therapist this week</p>
        <p>• Try the <button onclick="openCbtExercise()">5-4-3-2-1 grounding technique</button> when distressed</p>
      </div>`;
  }
  else {
    response += "Reflect on these alternative thoughts regularly to build mental resilience.";
  }
  
  addBotMessage(response);
  closeCbtExercise();
}

// ================== Text-to-Speech & Voice ==================
let isRecording = false;
const voiceBtn = document.getElementById('voice-btn');
const userInput = document.getElementById('user-input');

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  voiceBtn.addEventListener('click', () => {
    if (!isRecording) {
      recognition.start();
      voiceBtn.classList.add('recording');
    } else {
      recognition.stop();
      voiceBtn.classList.remove('recording');
    }
    isRecording = !isRecording;
  });

  recognition.onresult = e => {
    let transcript = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      transcript += e.results[i][0].transcript;
    }
    userInput.value = transcript;
  };

  recognition.onerror = e => {
    console.error('Speech recognition error:', e.error);
    voiceBtn.classList.remove('recording');
    isRecording = false;
  };
} else {
  voiceBtn.style.display = 'none';
}

function toggleResponseSpeech(button, text) {
  if (button.getAttribute('data-speaking') === 'true') {
    window.speechSynthesis.cancel();
    button.setAttribute('data-speaking', 'false');
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9;
  utterance.onend = () => button.setAttribute('data-speaking', 'false');
  window.speechSynthesis.speak(utterance);
  button.setAttribute('data-speaking', 'true');
}

// ================== Modal Outside Click ==================
window.onclick = function(event) {
  const bmiModal = document.getElementById('bmi-modal');
  const moodModal = document.getElementById('mood-modal');
  const cbtModal = document.getElementById('cbt-modal');
  if (event.target === bmiModal) closeBmiModal();
  if (event.target === moodModal) closeMoodTracker();
  if (event.target === cbtModal) closeCbtExercise();
};

// ================== Send Handlers ==================
document.getElementById('user-input').addEventListener('keypress', e => {
  if (e.key === 'Enter') sendMessage();
});
document.getElementById('send-btn').addEventListener('click', sendMessage);

// ===== Symptom Tracker stub =====
function showMedicationTracker() {
  alert("Symptom Tracker will be available in the next version.");
}
