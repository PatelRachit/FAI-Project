import React from "react";
import { Link } from "react-router-dom";

const NegativeResultPage = () => {
  return (
    <div className="min-h-screen bg-red-100 flex justify-center items-center">
      <div className="bg-white rounded-lg shadow-lg p-10 max-w-lg mx-auto text-center">
        <h2 className="text-4xl font-bold text-red-600 mb-6">
          Alert! Your Risk is High
        </h2>
        <p className="text-xl text-gray-700 mb-4">
          Based on your results, you are at high risk for diabetes. It's
          important to take action now.
        </p>
        <p className="text-lg text-gray-500 mb-8">
          Please consult a healthcare provider to discuss your results and
          develop a personalized plan to manage your health.
        </p>
        <Link
          to="/consultation"
          className="bg-red-600 text-white py-3 px-6 rounded-lg hover:bg-red-700 transition"
        >
          Get Professional Help
        </Link>
      </div>
    </div>
  );
};

export default NegativeResultPage;
