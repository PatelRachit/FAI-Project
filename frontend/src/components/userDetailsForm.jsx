import React, { useState } from "react";
import { useForm, Controller } from "react-hook-form";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormLabel from "@mui/material/FormLabel";
import TextField from "@mui/material/TextField";
import CircularProgress from "@mui/material/CircularProgress";
import Backdrop from "@mui/material/Backdrop";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Swal from "sweetalert2";
import { Accordion, AccordionSummary, AccordionDetails } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

const UserDetailsForm = () => {
  const {
    handleSubmit,
    control,
    reset,
    formState: { errors },
  } = useForm();

  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState("Health History"); // State for controlling accordion

  const onSubmit = async (formData) => {
    setLoading(true);
    formData.MentHlth = parseInt(formData.MentHlth) * 6;
    formData.PhysHlth = parseInt(formData.PhysHlth) * 6;
    try {
      const { data } = await axios.post(
        `${import.meta.env.VITE_SERVER_URL}/predict`,
        formData
      );
      setLoading(false);
      if (data.prediction) {
        Swal.fire({
          title: "Risk Identified",
          text: "You may be at risk for diabetes.",
          icon: "warning",
          confirmButtonColor: "#6366f1",
        });
      } else {
        Swal.fire({
          title: "All Clear",
          text: "You're currently not at risk for diabetes.",
          icon: "success",
          confirmButtonColor: "#10b981",
        });
      }
      reset();
      setExpanded(false);
    } catch (err) {
      setLoading(false);
      Swal.fire({
        title: "Error",
        text: "Something went wrong!",
        icon: "error",
        confirmButtonColor: "#ef4444",
      });
    }
  };

  const onReset = () => {
    reset();
    setExpanded(false); // Reset accordion state when resetting form
  };

  const descriptions = {
    GenHlth: [
      { value: "5", label: "Excellent" },
      { value: "4", label: "Very Good" },
      { value: "3", label: "Good" },
      { value: "2", label: "Fair" },
      { value: "1", label: "Poor" },
    ],
    MentHlth: [
      { value: "5", label: "Excellent" },
      { value: "4", label: "Very Good" },
      { value: "3", label: "Good" },
      { value: "2", label: "Fair" },
      { value: "1", label: "Poor" },
    ],
    PhysHlth: [
      { value: "5", label: "Excellent" },
      { value: "4", label: "Very Good" },
      { value: "3", label: "Good" },
      { value: "2", label: "Fair" },
      { value: "1", label: "Poor" },
    ],
    Education: [
      { value: "1", label: "Never attended school or only kindergarten" },
      { value: "2", label: "Grades 1 through 8" },
      { value: "3", label: "Grades 9 through 11" },
      { value: "4", label: "Grade 12 or GED" },
      { value: "5", label: "Some college or technical school" },
      { value: "6", label: "College 4 years or more" },
    ],
    Income: [
      { value: "1", label: "< $10,000" },
      { value: "2", label: "$10,000 to < $15,000" },
      { value: "3", label: "$15,000 to < $20,000" },
      { value: "4", label: "$20,000 to < $25,000" },
      { value: "5", label: "$25,000 to < $35,000" },
      { value: "6", label: "$35,000 to < $50,000" },
      { value: "7", label: "$50,000 to < $75,000" },
      { value: "8", label: "$75,000 or more" },
    ],
    Age: [
      { value: "1", label: "18-24" },
      { value: "2", label: "25-29" },
      { value: "3", label: "30-34" },
      { value: "4", label: "35-39" },
      { value: "5", label: "40-44" },
      { value: "6", label: "45-49" },
      { value: "7", label: "50-54" },
      { value: "8", label: "55-59" },
      { value: "9", label: "60-64" },
      { value: "10", label: "65-69" },
      { value: "11", label: "70-74" },
      { value: "12", label: "75-79" },
      { value: "13", label: "80 or older" },
    ],
  };

  const questions = [
    {
      group: "Health History",
      fields: [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
      ],
    },
    {
      group: "Demographics",
      fields: ["Sex", "Age", "Income", "Education"],
    },
    {
      group: "Health Ratings",
      fields: ["GenHlth", "MentHlth", "PhysHlth", "BMI"],
    },
  ];

  const fieldLabels = {
    HighBP: "Do you have high blood pressure?*",
    HighChol: "Do you have high cholesterol?*",
    CholCheck: "Have you had your cholesterol checked in the past 5 years?*",
    Smoker: "Have you smoked at least 100 cigarettes in your lifetime?*",
    Stroke: "Have you ever had a stroke?*",
    HeartDiseaseorAttack: "Have you had heart disease or a heart attack?*",
    PhysActivity: "Have you done any physical activity in the past 30 days?*",
    Fruits: "Do you eat fruits at least once per day?*",
    Veggies: "Do you eat vegetables at least once per day?*",
    HvyAlcoholConsump:
      "Do you consume alcohol heavily (more than 14 drinks/week for men, 7 for women)?*",
    AnyHealthcare: "Do you have any form of healthcare coverage?*",
    NoDocbcCost:
      "In the past year, was there a time you couldn't see a doctor because of cost?*",
    DiffWalk: "Do you have serious difficulty walking or climbing stairs?*",
    Sex: "What is your gender?*",
    Age: "What is your age category?",
    Income: "What is your income level?",
    Education: "What is your highest level of education?",
    GenHlth: "How would you rate your general health?",
    MentHlth: "How would you rate your mental health?",
    PhysHlth: "How would you rate your physical health?",
    BMI: "What is your Body Mass Index (BMI)?",
  };

  const getRadioOptions = (name) => {
    if (name === "Sex")
      return [
        { value: "1", label: "Male" },
        { value: "0", label: "Female" },
      ];
    if (descriptions[name]) return descriptions[name];
    return [
      { value: "1", label: "Yes" },
      { value: "0", label: "No" },
    ];
  };

  // This function will control accordion expansion based on errors
  const handleAccordionChange = (group) => (event, isExpanded) => {
    if (isExpanded) {
      setExpanded(group); // Only open the clicked accordion
    } else {
      setExpanded(false); // Close all accordions
    }
  };

  return (
    <section className="min-h-screen bg-gradient-to-br from-blue-200 via-blue-100 to-white py-12 px-4">
      <div className="max-w-4xl mx-auto bg-white shadow-2xl rounded-3xl p-10 relative">
        <h2 className="text-3xl font-bold text-center text-indigo-700 mb-4">
          Diabetes Risk Assessment
        </h2>
        <p className="text-sm text-center text-gray-600 mb-10">
          Know Your Risk, Take Control – Predict Diabetes Before It Strikes!
        </p>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-12">
          {questions.map(({ group, fields }) => (
            <Accordion
              key={group}
              defaultExpanded={group === "Health History"}
              expanded={expanded === group}
              onChange={handleAccordionChange(group)}
              sx={{
                marginBottom: "16px",
                borderRadius: "10px", // Rounded corners
                boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)", // Light shadow
                "&:before": {
                  display: "none",
                },
                "& .MuiAccordionSummary-root": {
                  padding: "0 24px",
                  background: "#fff", // Soft blue background
                  borderRadius: "10px", // Rounded corners
                  color: "#fff",
                },
                "& .MuiAccordionDetails-root": {
                  maxHeight: "400px", // Set max height for scrollable content
                  overflowY: "auto", // Enable scrolling
                  scrollbarWidth: "thin",
                  scrollbarColor: "#4C6EF5 #f1f5f9", // Custom scrollbar colors
                },
                "&::-webkit-scrollbar": {
                  width: "8px",
                },
                "&::-webkit-scrollbar-track": {
                  background: "#f1f5f9",
                },
                "&::-webkit-scrollbar-thumb": {
                  backgroundColor: "#4C6EF5",
                  borderRadius: "10px",
                },
              }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <h3 className="text-lg font-semibold text-indigo-700">
                  {group}
                </h3>
              </AccordionSummary>
              <AccordionDetails>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 ml-4">
                  {fields.map((name) => (
                    <div key={name}>
                      <FormLabel className="text-sm">
                        {fieldLabels[name]}
                      </FormLabel>
                      <Controller
                        name={name}
                        control={control}
                        defaultValue=""
                        rules={{ required: "This field is required" }}
                        render={({ field }) => {
                          if (name === "BMI") {
                            return (
                              <TextField
                                fullWidth
                                variant="outlined"
                                label="BMI"
                                {...field}
                                error={!!errors[name]}
                                helperText={errors[name]?.message || ""}
                                size="small" // Reduced size
                              />
                            );
                          }
                          const options = getRadioOptions(name);
                          return (
                            <RadioGroup
                              row
                              {...field}
                              sx={{ display: "flex", flexDirection: "column" }}
                            >
                              {options.map((opt) => (
                                <FormControlLabel
                                  key={opt.value || opt}
                                  value={opt.value || opt}
                                  control={<Radio />}
                                  label={opt.label || opt}
                                />
                              ))}
                            </RadioGroup>
                          );
                        }}
                      />
                      {errors[name] && (
                        <p className="text-red-500 text-xs">
                          {errors[name]?.message}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </AccordionDetails>
            </Accordion>
          ))}
          <div className="flex justify-center gap-6">
            <button
              type="submit"
              className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 font-semibold shadow-md text-lg cursor-pointer"
            >
              Submit
            </button>
            <button
              type="button"
              onClick={onReset}
              className="bg-gray-400 text-white px-6 py-2 rounded-lg hover:bg-gray-500 shadow-md text-lg font-semibold cursor-pointer"
            >
              Reset
            </button>
          </div>
        </form>

        <Backdrop
          sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
          open={loading}
        >
          <CircularProgress color="inherit" />
          <br />
          <p className="text-white mt-4 text-lg font-medium">
            Analyzing your responses... This might take a few minutes.
          </p>
        </Backdrop>
      </div>
    </section>
  );
};

export default UserDetailsForm;
