import React from "react";

function SosButton() {
  const triggerSOS = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/send-sos", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          location: "https://maps.google.com/?q=12.9716,77.5946",
          message: "Feeling unsafe!",
        }),
      });

      const result = await response.json();
      if (result.success) {
        alert("🚨 SOS alert sent successfully!");
      } else {
        alert("❌ Failed to send SOS alert.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("⚠️ Error sending SOS alert.");
    }
  };

  return (
    <button
      onClick={triggerSOS}
      className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
    >
      Send SOS
    </button>
  );
}

export default SosButton;
