import React from "react";
import { Link } from "react-router-dom";

const PositiveResultPage = () => {
  return (
    <div className="min-h-screen bg-green-100 flex justify-center items-center">
      <div className="bg-white rounded-lg shadow-lg p-10 max-w-lg mx-auto text-center">
        <h2 className="text-4xl font-bold text-green-600 mb-6">
          Congratulations! Your Risk is Low
        </h2>
        <p className="text-xl text-gray-700 mb-4">
          Your results show that your risk for diabetes is low. Keep up the
          great work with your healthy habits!
        </p>
        <p className="text-lg text-gray-500 mb-8">
          Continue to stay active, eat a balanced diet, and monitor your health
          regularly.
        </p>
        <Link
          to="/"
          className="bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 transition"
        >
          Take another test
        </Link>
      </div>
    </div>
  );
};

export default PositiveResultPage;
