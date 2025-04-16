import { useState } from "react";
import axios from "axios";
import { BrowserRouter as Router } from "react-router-dom";
import RoutesProvider from "./routes";

function App() {
  const [count, setCount] = useState(0);

  const checkApiCall = async () => {
    const { data } = await axios.get(
      `${import.meta.env.VITE_SERVER_URL}/predict`
    );
    console.log(data);
  };
  return (
    <>
      <Router>
        <RoutesProvider />
      </Router>
    </>
  );
}

export default App;
