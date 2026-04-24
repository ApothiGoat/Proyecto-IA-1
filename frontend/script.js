function generateTicketId() {
  return "TCK-" + Math.floor(Math.random() * 100000);
}

document.getElementById("ticketId").innerText = generateTicketId();

async function sendTicket() {
  const subject = document.getElementById("subject").value;
  const description = document.getElementById("description").value;

  const text = subject + " " + description;

  if (!text.trim()) {
    alert("Completa el ticket");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    document.getElementById("prediction").innerText = data.prediction;

    // Mostrar barra de confianza
    const confidence = data.confidence * 100;

    document.getElementById("confidenceBar").style.width = confidence + "%";
    document.getElementById("confidenceText").innerText = confidence + "%";

    document.getElementById("result").classList.remove("hidden");

  } catch (error) {
    alert("Error conectando con backend");
  }
}