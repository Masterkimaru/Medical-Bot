document.getElementById("chat-form").addEventListener("submit", function(e) {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const messageText = input.value.trim();
  if (messageText !== "") {
    appendMessage("user", messageText);
    // Here you would send the message to your backend and then append the bot's response.
    input.value = "";
  }
});

function appendMessage(sender, text) {
  const chatWindow = document.getElementById("chat-window");
  const messageElem = document.createElement("div");
  messageElem.classList.add("message", sender);
  const textElem = document.createElement("div");
  textElem.classList.add("text");
  textElem.textContent = text;
  messageElem.appendChild(textElem);
  chatWindow.appendChild(messageElem);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
