import React from 'react';
import UploadForm from './components/UploadForm';
import './styles.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Multi-Modal Skin Pigmentation Detection System</h1>
      </header>
      <main>
        <UploadForm />
      </main>
    </div>
  );
}

export default App;