function generateTicketId() {
  return "TCK-" + Math.floor(10000 + Math.random() * 90000);
}

document.getElementById("ticketId").innerText = generateTicketId();

async function sendTicket() {
  const subject = document.getElementById("subject").value.trim();
  const description = document.getElementById("description").value.trim();
  const button = document.getElementById("classifyBtn");

  const text = `${subject} ${description}`.trim();

  if (!text) {
    alert("Completa el subject o la descripción del ticket.");
    return;
  }

  try {
    button.innerText = "Clasificando...";
    button.disabled = true;

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    document.getElementById("prediction").innerText = data.prediction;

    const confidence = Math.round(data.confidence * 100);
    document.getElementById("confidenceBar").style.width = confidence + "%";
    document.getElementById("confidenceText").innerText = confidence + "%";

    const probabilitiesDiv = document.getElementById("probabilities");
    probabilitiesDiv.innerHTML = "";

    const sortedProbabilities = Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1]);

    sortedProbabilities.forEach(([category, probability]) => {
      const percent = Math.round(probability * 100);

      probabilitiesDiv.innerHTML += `
        <div class="prob-row">
          <div class="prob-info">
            <span>${category}</span>
            <span>${percent}%</span>
          </div>
          <div class="bar">
            <div class="prob-bar" style="width: ${percent}%"></div>
          </div>
        </div>
      `;
    });

    document.getElementById("result").classList.remove("hidden");

  } catch (error) {
    alert("No se pudo conectar con el backend.");
    console.error(error);
  } finally {
    button.innerText = "Clasificar Ticket";
    button.disabled = false;
  }
}