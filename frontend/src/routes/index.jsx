import { Route, Routes } from "react-router-dom";
import { NagetiveDiabetes, PositiveDiabetes, UserDetails } from "../pages";

const RoutesProvider = () => {
  return (
    <Routes>
      <Route path="/" element={<UserDetails />} />
      <Route path="/positive" element={<PositiveDiabetes />} />
      <Route path="/nagetive" element={<NagetiveDiabetes />} />
    </Routes>
  );
};

export default RoutesProvider;
