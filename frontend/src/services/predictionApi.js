async function parseResponse(response) {
  let data = null;

  try {
    data = await response.json();
  } catch {
    // Ignore JSON parsing failure.
  }


  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`;


    if (Array.isArray(data?.detail)) {
      message = data.detail
        .map((item) => {
          const field =
            item.loc?.at(-1)
            ?? "request";

          return `${field}: ${item.msg}`;
        })
        .join(" | ");
    } else if (
      typeof data?.detail === "string"
    ) {
      message =
        data.detail;
    } else if (data) {
      message =
        JSON.stringify(data);
    }


    throw new Error(
      message
    );
  }


  return data;
}


export async function predictReadmission(
  payload
) {
  const response =
    await fetch(
      "/api/predict",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            payload
          ),
      }
    );


  return parseResponse(
    response
  );
}


export async function checkReady() {
  const response =
    await fetch(
      "/api/ready"
    );


  return parseResponse(
    response
  );
}