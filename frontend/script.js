document
    .getElementById("predictionForm")
    .addEventListener("submit", async function(event) {

        event.preventDefault();


        const button =
            document.getElementById("predictButton");

        const buttonText =
            document.getElementById("buttonText");

        const resultBox =
            document.getElementById("result");


        // Clear previous result

        resultBox.className = "";

        resultBox.innerText = "";


        // Show loading animation

        button.classList.add("loading");

        buttonText.innerText = "Predicting...";


        const data = {

            Pregnancies:
                Number(
                    document.getElementById("Pregnancies").value
                ),

            Glucose:
                Number(
                    document.getElementById("Glucose").value
                ),

            BloodPressure:
                Number(
                    document.getElementById("BloodPressure").value
                ),

            SkinThickness:
                Number(
                    document.getElementById("SkinThickness").value
                ),

            Insulin:
                Number(
                    document.getElementById("Insulin").value
                ),

            BMI:
                Number(
                    document.getElementById("BMI").value
                ),

            DiabetesPedigreeFunction:
                Number(
                    document
                        .getElementById(
                            "DiabetesPedigreeFunction"
                        )
                        .value
                ),

            Age:
                Number(
                    document.getElementById("Age").value
                )
        };


        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    result.error ||
                    "Prediction failed"
                );
            }


            // Small delay so loading animation is visible

            await new Promise(
                resolve =>
                    setTimeout(resolve, 500)
            );


            if (result.prediction === 1) {

                resultBox.innerText =
                    "⚠️ Diabetes Risk Detected";

                resultBox.className =
                    "danger";

            } else {

                resultBox.innerText =
                    "✅ No Diabetes Risk";

                resultBox.className =
                    "success";
            }


        } catch (error) {

            console.error(error);


            resultBox.innerText =
                "❌ Error connecting to Flask API.";

            resultBox.className =
                "error";

        } finally {

            button.classList.remove("loading");

            buttonText.innerText =
                "Predict Diabetes Risk";
        }

    });