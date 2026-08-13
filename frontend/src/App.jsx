import {
  useEffect,
  useState,
} from "react";
import {
  predictReadmission,
} from "./services/predictionApi";

// ============================================================
// MEDICATION FIELDS
// ============================================================

const medicationFields = [
  "metformin",
  "repaglinide",
  "nateglinide",
  "chlorpropamide",
  "glimepiride",
  "glipizide",
  "glyburide",
  "pioglitazone",
  "rosiglitazone",
  "acarbose",
  "miglitol",
  "insulin",
  "glyburide-metformin",
  "tolazamide",
  "metformin-pioglitazone",
  "metformin-rosiglitazone",
  "glimepiride-pioglitazone",
  "glipizide-metformin",
  "troglitazone",
  "tolbutamide",
  "acetohexamide",
];


// ============================================================
// INITIAL FORM
// ============================================================

const initialForm = {
  race: "",
  gender: "",
  age: "",

  admission_type_id: "",
  discharge_disposition_id: "",
  admission_source_id: "",
  time_in_hospital: "",

  num_lab_procedures: "",
  num_procedures: "",
  num_medications: "",
  number_diagnoses: "",

  number_outpatient: "",
  number_emergency: "",
  number_inpatient: "",

  max_glu_serum: "",
  A1Cresult: "",
  diag_1: "",

  change: "",
  diabetesMed: "",

  ...Object.fromEntries(
    medicationFields.map(
      (name) => [
        name,
        "",
      ]
    )
  ),
};

const sampleRequest = {
  race: "Caucasian",
  gender: "Female",
  age: "[80-90)",

  admission_type_id: "1",
  discharge_disposition_id: "1",
  admission_source_id: "7",

  time_in_hospital: "10",
  num_lab_procedures: "70",
  num_procedures: "2",
  num_medications: "25",

  number_outpatient: "3",
  number_emergency: "2",
  number_inpatient: "4",
  number_diagnoses: "9",

  max_glu_serum: ">300",
  A1Cresult: ">8",

  metformin: "No",
  repaglinide: "No",
  nateglinide: "No",
  chlorpropamide: "No",
  glimepiride: "No",
  glipizide: "No",
  glyburide: "No",
  pioglitazone: "No",
  rosiglitazone: "No",
  acarbose: "No",
  miglitol: "No",
  insulin: "Up",
  "glyburide-metformin": "No",
  tolazamide: "No",
  "metformin-pioglitazone": "No",
  "metformin-rosiglitazone": "No",
  "glimepiride-pioglitazone": "No",
  "glipizide-metformin": "No",
  troglitazone: "No",
  tolbutamide: "No",
  acetohexamide: "No",

  change: "Ch",
  diabetesMed: "Yes",

  diag_1: "250.13",
};

// ============================================================
// NUMERIC FIELDS
// ============================================================

const numericFields = [
  "admission_type_id",
  "discharge_disposition_id",
  "admission_source_id",
  "time_in_hospital",
  "num_lab_procedures",
  "num_procedures",
  "num_medications",
  "number_outpatient",
  "number_emergency",
  "number_inpatient",
  "number_diagnoses",
];


// ============================================================
// APP
// ============================================================

function App() {
  const [form, setForm] =
    useState(initialForm);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState(null);

  const [
    activeSection,
    setActiveSection,
  ] = useState("patient-info");

  const loadExample = () => {
  setForm({
    ...sampleRequest,
  });

  setResult(null);
  setError(null);

  setActiveSection(
    "patient-info"
  );

  window.location.hash =
    "patient-info";
};

  // ==========================================================
  // SIDEBAR ACTIVE SECTION
  // ==========================================================

  useEffect(() => {
    const sectionIds = [
      "patient-info",
      "admission",
      "clinical",
      "utilization",
      "labs",
      "treatment",
      "medications",
    ];


    const observer =
      new IntersectionObserver(
        (entries) => {
          const visibleSections =
            entries
              .filter(
                (entry) =>
                  entry.isIntersecting
              )
              .sort(
                (a, b) =>
                  a.boundingClientRect.top
                  - b.boundingClientRect.top
              );


          if (
            visibleSections.length > 0
          ) {
            setActiveSection(
              visibleSections[0]
                .target
                .id
            );
          }
        },
        {
          root: null,
          rootMargin:
            "-15% 0px -70% 0px",
          threshold: 0,
        }
      );


    sectionIds.forEach(
      (id) => {
        const element =
          document.getElementById(
            id
          );

        if (element) {
          observer.observe(
            element
          );
        }
      }
    );


    return () => {
      observer.disconnect();
    };
  }, []);


  // ==========================================================
  // FORM EVENTS
  // ==========================================================

  const updateField = (
    event
  ) => {
    const {
      name,
      value,
    } = event.target;


    setForm(
      (current) => ({
        ...current,
        [name]: value,
      })
    );


    setError(null);
  };


  const resetForm = () => {
    setForm(initialForm);

    setResult(null);

    setError(null);

    setActiveSection(
      "patient-info"
    );


    window.location.hash =
      "patient-info";
  };


  // ==========================================================
  // SIDEBAR CLICK
  // ==========================================================

  const handleSidebarClick = (
    sectionId
  ) => {
    setActiveSection(
      sectionId
    );


    if (
      sectionId
      === "medications"
    ) {
      const medicationSection =
        document.getElementById(
          "medications"
        );


      if (
        medicationSection
        && medicationSection
          .tagName
          .toLowerCase()
          === "details"
      ) {
        medicationSection.open =
          true;
      }
    }
  };


  // ==========================================================
  // VALIDATE FORM
  // ==========================================================

  const handlePredict = async () => {
  console.log(
    "FORM DATA:",
    form
  );


  // ----------------------------------------------------------
  // 1. Check required fields
  // ----------------------------------------------------------

  const missingFields =
    Object.entries(form)
      .filter(
        ([, value]) =>
          value === ""
          || value === null
          || value === undefined
      )
      .map(
        ([field]) => field
      );


  if (missingFields.length > 0) {
    setError(
      `Missing fields: ${missingFields.join(", ")}`
    );

    return;
  }


  // ----------------------------------------------------------
  // 2. Build payload
  // ----------------------------------------------------------

  const payload = {
    ...form,
  };


  numericFields.forEach(
    (field) => {
      payload[field] =
        Number(
          payload[field]
        );
    }
  );


  console.log(
    "FINAL PAYLOAD:",
    payload
  );

  console.log(
    "Payload field count:",
    Object.keys(payload).length
  );


  // ----------------------------------------------------------
  // 3. Call backend
  // ----------------------------------------------------------

  try {
    setLoading(true);
    setError(null);
    setResult(null);


    const prediction =
      await predictReadmission(
        payload
      );


    console.log(
      "PREDICTION RESPONSE:",
      prediction
    );


    setResult(
      prediction
    );

  } catch (requestError) {

    console.error(
      "Prediction error:",
      requestError
    );


    setError(
      requestError.message
      || "Unable to calculate prediction."
    );

  } finally {

    setLoading(false);

  }
};


  return (
    <div
      className="
        min-h-screen
        bg-[#f7f9fb]
        text-[#191c1e]
      "
    >

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header
        className="
          sticky
          top-0
          z-50

          h-16

          border-b
          border-[#bfc7d2]

          bg-white/95
          backdrop-blur
        "
      >
        <div
          className="
            h-full

            px-5
            md:px-10

            flex
            items-center
            justify-between
          "
        >
          <h1
            className="
              text-lg
              md:text-xl

              font-bold

              text-[#006194]
            "
          >
            Patient Readmission Risk Assessment
          </h1>


          <div
            className="
              hidden
              sm:block

              text-sm
              font-semibold

              text-[#006a61]
            "
          >
            AI Decision Support
          </div>
        </div>
      </header>


      <div className="flex">

        {/* ====================================================
            SIDEBAR
        ==================================================== */}

        <aside
          className="
            hidden
            md:flex

            fixed

            top-16
            left-0

            w-64
            h-[calc(100vh-64px)]

            bg-[#f2f4f6]

            border-r
            border-[#bfc7d2]

            flex-col

            p-4
          "
        >

          <div className="mb-6">

            <div
              className="
                text-sm
                font-bold
                text-[#006a61]
              "
            >
              AI Decision Support
            </div>


            <div
              className="
                text-xs
                text-gray-500
              "
            >
              Patient Readmission Model
            </div>

          </div>


          <nav
            className="
              flex
              flex-col
              gap-2
              text-sm
            "
          >

            <SidebarLink
              href="#patient-info"
              label="Patient Info"
              active={
                activeSection
                === "patient-info"
              }
              onClick={() =>
                handleSidebarClick(
                  "patient-info"
                )
              }
            />


            <SidebarLink
              href="#admission"
              label="Admission Details"
              active={
                activeSection
                === "admission"
              }
              onClick={() =>
                handleSidebarClick(
                  "admission"
                )
              }
            />


            <SidebarLink
              href="#clinical"
              label="Clinical Activity"
              active={
                activeSection
                === "clinical"
              }
              onClick={() =>
                handleSidebarClick(
                  "clinical"
                )
              }
            />


            <SidebarLink
              href="#utilization"
              label="Prior Utilization"
              active={
                activeSection
                === "utilization"
              }
              onClick={() =>
                handleSidebarClick(
                  "utilization"
                )
              }
            />


            <SidebarLink
              href="#labs"
              label="Labs / Diagnosis"
              active={
                activeSection
                === "labs"
              }
              onClick={() =>
                handleSidebarClick(
                  "labs"
                )
              }
            />


            <SidebarLink
              href="#treatment"
              label="Treatment"
              active={
                activeSection
                === "treatment"
              }
              onClick={() =>
                handleSidebarClick(
                  "treatment"
                )
              }
            />


            <SidebarLink
              href="#medications"
              label="Diabetes Meds"
              active={
                activeSection
                === "medications"
              }
              onClick={() =>
                handleSidebarClick(
                  "medications"
                )
              }
            />

          </nav>

        </aside>


        {/* ====================================================
            MAIN
        ==================================================== */}

        <main
          className="
            flex-1

            md:ml-64

            px-5
            md:px-10

            py-8
          "
        >

          {/* ==================================================
              INTRO
          ================================================== */}

          <div className="mb-8">

            <span
              className="
                inline-flex

                px-3
                py-1

                rounded-full

                bg-[#cce5ff]
                text-[#004b73]

                text-xs
                font-bold
              "
            >
              AI Decision Support
            </span>


            <h2
              className="
                text-2xl
                md:text-3xl

                font-bold

                mt-3
              "
            >
              Patient Readmission Risk Assessment
            </h2>


            <p
              className="
                mt-2

                text-gray-600
                text-base
                md:text-lg
              "
            >
              Estimate the risk of inpatient
              readmission within 30 days after
              discharge.
            </p>


            <div
              className="
                mt-4

                max-w-3xl

                bg-white

                border
                border-[#bfc7d2]

                rounded-lg

                p-3

                text-sm
                text-gray-600
              "
            >
              This tool supports clinical
              decision-making and does not
              replace professional medical
              judgment.
            </div>

          </div>


          {/* ==================================================
              CONTENT GRID
          ================================================== */}

          <div
            className="
              grid
              grid-cols-1
              lg:grid-cols-12

              gap-8

              items-start
            "
          >

            {/* ================================================
                FORM COLUMN
            ================================================ */}

            <div
              className="
                lg:col-span-8

                flex
                flex-col

                gap-6
              "
            >

              {/* PATIENT */}

              <FormSection
                id="patient-info"
                title="Patient Information"
              >

                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-3
                    gap-5
                  "
                >

                  <SelectField
                    label="Race"
                    name="race"
                    value={
                      form.race
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      {
                        value:
                          "Caucasian",
                        label:
                          "Caucasian",
                      },
                      {
                        value:
                          "AfricanAmerican",
                        label:
                          "African American",
                      },
                      {
                        value:
                          "Asian",
                        label:
                          "Asian",
                      },
                      {
                        value:
                          "Hispanic",
                        label:
                          "Hispanic",
                      },
                      {
                        value:
                          "Other",
                        label:
                          "Other",
                      },
                    ]}
                  />


                  <SelectField
                    label="Gender"
                    name="gender"
                    value={
                      form.gender
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      "Female",
                      "Male",
                    ]}
                  />


                  <SelectField
                    label="Age Group"
                    name="age"
                    value={
                      form.age
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      "[0-10)",
                      "[10-20)",
                      "[20-30)",
                      "[30-40)",
                      "[40-50)",
                      "[50-60)",
                      "[60-70)",
                      "[70-80)",
                      "[80-90)",
                      "[90-100)",
                    ]}
                  />

                </div>

              </FormSection>


              {/* ADMISSION */}

              <FormSection
                id="admission"
                title="Admission & Discharge"
              >

                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-2
                    gap-5
                  "
                >

                  <NumberField
                    label="Admission Type ID"
                    name="admission_type_id"
                    value={
                      form.admission_type_id
                    }
                    onChange={
                      updateField
                    }
                    min="1"
                  />


                  <NumberField
                    label="Admission Source ID"
                    name="admission_source_id"
                    value={
                      form.admission_source_id
                    }
                    onChange={
                      updateField
                    }
                    min="1"
                  />


                  <NumberField
                    label="Discharge Disposition ID"
                    name="discharge_disposition_id"
                    value={
                      form.discharge_disposition_id
                    }
                    onChange={
                      updateField
                    }
                    min="1"
                  />


                  <NumberField
                    label="Time in Hospital"
                    name="time_in_hospital"
                    value={
                      form.time_in_hospital
                    }
                    onChange={
                      updateField
                    }
                    min="1"
                    suffix="days"
                  />

                </div>

              </FormSection>


              {/* CLINICAL */}

              <FormSection
                id="clinical"
                title="Clinical Activity"
              >

                <div
                  className="
                    grid
                    grid-cols-1
                    sm:grid-cols-2
                    xl:grid-cols-4
                    gap-5
                  "
                >

                  <NumberField
                    label="Lab Procedures"
                    name="num_lab_procedures"
                    value={
                      form.num_lab_procedures
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />


                  <NumberField
                    label="Other Procedures"
                    name="num_procedures"
                    value={
                      form.num_procedures
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />


                  <NumberField
                    label="Medications"
                    name="num_medications"
                    value={
                      form.num_medications
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />


                  <NumberField
                    label="Diagnoses"
                    name="number_diagnoses"
                    value={
                      form.number_diagnoses
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />

                </div>

              </FormSection>


              {/* UTILIZATION */}

              <FormSection
                id="utilization"
                title="Prior Healthcare Utilization"
              >

                <p
                  className="
                    text-sm
                    text-gray-500
                    mb-5
                  "
                >
                  Recorded visits during the year
                  preceding the current encounter.
                </p>


                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-3
                    gap-5
                  "
                >

                  <NumberField
                    label="Outpatient Visits"
                    name="number_outpatient"
                    value={
                      form.number_outpatient
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />


                  <NumberField
                    label="Emergency Visits"
                    name="number_emergency"
                    value={
                      form.number_emergency
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />


                  <NumberField
                    label="Inpatient Visits"
                    name="number_inpatient"
                    value={
                      form.number_inpatient
                    }
                    onChange={
                      updateField
                    }
                    min="0"
                  />

                </div>

              </FormSection>


              {/* LABS */}

              <FormSection
                id="labs"
                title="Laboratory & Diagnosis"
              >

                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-3
                    gap-5
                  "
                >

                  <SelectField
                    label="Max Glucose Serum"
                    name="max_glu_serum"
                    value={
                      form.max_glu_serum
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      {
                        value:
                          "None",
                        label:
                          "Not measured",
                      },
                      {
                        value:
                          "Norm",
                        label:
                          "Normal",
                      },
                      ">200",
                      ">300",
                    ]}
                  />


                  <SelectField
                    label="A1C Result"
                    name="A1Cresult"
                    value={
                      form.A1Cresult
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      {
                        value:
                          "None",
                        label:
                          "Not measured",
                      },
                      {
                        value:
                          "Norm",
                        label:
                          "Normal",
                      },
                      ">7",
                      ">8",
                    ]}
                  />


                  <TextField
                    label="Primary Diagnosis (ICD-9)"
                    name="diag_1"
                    value={
                      form.diag_1
                    }
                    onChange={
                      updateField
                    }
                    placeholder="e.g. 250.13"
                  />

                </div>

              </FormSection>


              {/* TREATMENT */}

              <FormSection
                id="treatment"
                title="Diabetes Treatment"
              >

                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-2
                    gap-5
                  "
                >

                  <SelectField
                    label="Diabetes Medication Changed"
                    name="change"
                    value={
                      form.change
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      {
                        value:
                          "Ch",
                        label:
                          "Changed",
                      },
                      {
                        value:
                          "No",
                        label:
                          "No Change",
                      },
                    ]}
                  />


                  <SelectField
                    label="Diabetes Medication Prescribed"
                    name="diabetesMed"
                    value={
                      form.diabetesMed
                    }
                    onChange={
                      updateField
                    }
                    options={[
                      "Yes",
                      "No",
                    ]}
                  />

                </div>

              </FormSection>


              {/* MEDICATIONS */}

              <details
                id="medications"
                className="
                  scroll-mt-24

                  bg-white

                  rounded-xl

                  border
                  border-[#bfc7d2]

                  p-6

                  shadow-sm
                "
              >

                <summary
                  className="
                    cursor-pointer
                    text-xl
                    font-semibold
                  "
                >
                  Medication Details
                </summary>


                <p
                  className="
                    text-sm
                    text-gray-500
                    mt-3
                    mb-5
                  "
                >
                  Medication status recorded during
                  the current encounter.
                </p>


                <div
                  className="
                    grid
                    grid-cols-1
                    md:grid-cols-2
                    gap-4
                  "
                >

                  {medicationFields.map(
                    (
                      medication
                    ) => (
                      <SelectField
                        key={
                          medication
                        }
                        label={
                          formatMedicationName(
                            medication
                          )
                        }
                        name={
                          medication
                        }
                        value={
                          form[
                            medication
                          ]
                        }
                        onChange={
                          updateField
                        }
                        options={[
                          {
                            value:
                              "No",
                            label:
                              "Not Prescribed",
                          },
                          {
                            value:
                              "Steady",
                            label:
                              "Steady",
                          },
                          {
                            value:
                              "Up",
                            label:
                              "Dose Increased",
                          },
                          {
                            value:
                              "Down",
                            label:
                              "Dose Decreased",
                          },
                        ]}
                      />
                    )
                  )}

                </div>

              </details>


              {/* ACTIONS */}

              <div
                className="
                  flex
                  flex-wrap
                  gap-3
                  pb-10
                "
              >

                <button
                  type="button"
                  onClick={
                    handlePredict
                  }
                  disabled={
                    loading
                  }
                  className="
                    px-6
                    py-3

                    rounded-lg

                    bg-[#006194]
                    text-white

                    font-semibold

                    hover:bg-[#004b73]

                    disabled:opacity-50
                    disabled:cursor-not-allowed
                  "
                >
                  {loading
                    ? "Calculating..."
                    : "Predict Readmission Risk"}
                </button>

                <button
                type="button"
                onClick={loadExample}
                disabled={loading}
                className="
                  px-6
                  py-3

                  rounded-lg

                  border
                  border-[#006194]

                  bg-white
                  text-[#006194]

                  font-semibold

                  hover:bg-[#eef8ff]

                  disabled:opacity-50
                "
              >
                Load Example
              </button>


                <button
                  type="button"
                  onClick={
                    resetForm
                  }
                  disabled={
                    loading
                  }
                  className="
                    px-6
                    py-3

                    rounded-lg

                    border
                    border-[#bfc7d2]

                    bg-white

                    font-semibold

                    hover:bg-gray-100

                    disabled:opacity-50
                  "
                >
                  Reset
                </button>

              </div>

            </div>


            {/* ================================================
                RESULT PANEL
            ================================================ */}

            <aside
              className="
                lg:col-span-4

                lg:sticky
                lg:top-24
              "
            >

              <div
                className="
                  bg-white

                  border
                  border-[#bfc7d2]

                  rounded-xl

                  p-6

                  shadow-md
                "
              >

                <h2
                  className="
                    text-xl
                    font-semibold

                    border-b
                    border-gray-300

                    pb-3
                  "
                >
                  Prediction Results
                </h2>


                {!result
                  && !loading
                  && (
                    <div
                      className="
                        py-10

                        text-center
                        text-gray-500
                      "
                    >
                      Complete the clinical
                      information and submit
                      the form to calculate risk.
                    </div>
                  )}


                {loading && (
                  <div
                    className="
                      py-10
                      text-center
                    "
                  >
                    Calculating readmission
                    risk...
                  </div>
                )}


                {error && (
                  <div
                    className="
                      mt-4

                      rounded-lg

                      bg-red-50
                      text-red-700

                      p-3

                      text-sm

                      break-words
                    "
                  >
                    {error}
                  </div>
                )}


                {result && (
                  <PredictionResult
                    result={
                      result
                    }
                  />
                )}

              </div>

            </aside>

          </div>

        </main>

      </div>

    </div>
  );
}


// ============================================================
// REUSABLE COMPONENTS
// ============================================================

function SidebarLink({
  href,
  label,
  active = false,
  onClick,
}) {
  return (
    <a
      href={href}
      onClick={
        onClick
      }
      aria-current={
        active
          ? "location"
          : undefined
      }
      className={`
        p-3

        rounded-lg

        transition-colors

        ${
          active
            ? `
              bg-[#86f2e4]
              text-[#006f66]
              font-semibold
            `
            : `
              text-[#191c1e]
              hover:bg-gray-200
            `
        }
      `}
    >
      {label}
    </a>
  );
}


function FormSection({
  id,
  title,
  children,
}) {
  return (
    <section
      id={id}
      className="
        scroll-mt-24

        bg-white

        rounded-xl

        border
        border-[#bfc7d2]

        p-6

        shadow-sm
      "
    >

      <SectionTitle>
        {title}
      </SectionTitle>

      {children}

    </section>
  );
}


function SectionTitle({
  children,
}) {
  return (
    <h2
      className="
        text-xl
        font-semibold

        border-b
        border-gray-200

        pb-3
        mb-5
      "
    >
      {children}
    </h2>
  );
}


function SelectField({
  label,
  name,
  value,
  onChange,
  options,
}) {
  return (
    <label
      className="
        flex
        flex-col
        gap-2
      "
    >

      <span
        className="
          text-sm
          font-semibold
        "
      >
        {label}
      </span>


      <select
        name={
          name
        }
        value={
          value
        }
        onChange={
          onChange
        }
        className="
          h-11

          rounded-lg

          border
          border-[#bfc7d2]

          bg-white

          px-3

          focus:outline-none
          focus:ring-2
          focus:ring-[#93ccff]
        "
      >

        <option value="">
          Select...
        </option>


        {options.map(
          (option) => {
            const normalized =
              typeof option
                === "string"
                ? {
                    value:
                      option,
                    label:
                      option,
                  }
                : option;


            return (
              <option
                key={
                  normalized
                    .value
                }
                value={
                  normalized
                    .value
                }
              >
                {
                  normalized
                    .label
                }
              </option>
            );
          }
        )}

      </select>

    </label>
  );
}


function NumberField({
  label,
  name,
  value,
  onChange,
  min,
  max,
  suffix,
}) {
  return (
    <label
      className="
        flex
        flex-col
        gap-2
      "
    >

      <span
        className="
          text-sm
          font-semibold
        "
      >
        {label}
      </span>


      <div
        className="
          relative
        "
      >

        <input
          type="number"
          name={
            name
          }
          value={
            value
          }
          onChange={
            onChange
          }
          min={
            min
          }
          max={
            max
          }
          placeholder="Enter value"
          className={`
            h-11
            w-full

            rounded-lg

            border
            border-[#bfc7d2]

            bg-white

            px-3

            ${
              suffix
                ? "pr-14"
                : ""
            }

            focus:outline-none
            focus:ring-2
            focus:ring-[#93ccff]
          `}
        />


        {suffix && (
          <span
            className="
              absolute

              right-3
              top-1/2

              -translate-y-1/2

              text-sm
              text-gray-500

              pointer-events-none
            "
          >
            {suffix}
          </span>
        )}

      </div>

    </label>
  );
}


function TextField({
  label,
  name,
  value,
  onChange,
  placeholder,
}) {
  return (
    <label
      className="
        flex
        flex-col
        gap-2
      "
    >

      <span
        className="
          text-sm
          font-semibold
        "
      >
        {label}
      </span>


      <input
        type="text"
        name={
          name
        }
        value={
          value
        }
        onChange={
          onChange
        }
        placeholder={
          placeholder
        }
        className="
          h-11

          rounded-lg

          border
          border-[#bfc7d2]

          bg-white

          px-3

          focus:outline-none
          focus:ring-2
          focus:ring-[#93ccff]
        "
      />

    </label>
  );
}


function PredictionResult({
  result,
}) {
  return (
    <div className="mt-6">

      <div
        className="
          text-sm
          uppercase
          tracking-wide
          text-gray-500
        "
      >
        30-Day Readmission Risk
      </div>


      <div
        className="
          text-center

          text-5xl
          md:text-6xl

          font-extrabold

          text-[#006194]

          my-6
        "
      >
        {(
          result
            .risk_score
          * 100
        ).toFixed(1)}
        %
      </div>


      <div
        className="
          text-center
        "
      >

        <span
          className={`
            inline-flex

            rounded-full

            px-4
            py-2

            font-bold

            ${
              result
                .prediction
                === 1
                ? `
                  bg-red-100
                  text-red-800
                `
                : `
                  bg-emerald-100
                  text-emerald-800
                `
            }
          `}
        >
          {result
            .prediction
            === 1
            ? "HIGH RISK"
            : "NOT HIGH RISK"}
        </span>

      </div>


      {result.status && (
        <p
          className="
            mt-5

            text-center
            text-sm
            text-gray-600
          "
        >
          Status:{" "}
          <strong>
            {
              result
                .status
            }
          </strong>
        </p>
      )}


      <div
        className="
          mt-6

          border-t
          border-gray-200

          pt-4

          text-xs
          text-gray-500
        "
      >
        This prediction provides clinical
        decision support only and should be
        interpreted together with the
        patient's clinical context.
      </div>

    </div>
  );
}


// ============================================================
// HELPERS
// ============================================================

function formatMedicationName(
  medication
) {
  return medication
    .split("-")
    .map(
      (word) =>
        word
          .charAt(0)
          .toUpperCase()
        + word.slice(1)
    )
    .join(" ");
}


export default App;