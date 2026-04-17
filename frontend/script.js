/**
 * Tab Switching Logic
 */
function showTab(tabName) {
  const tabs = document.querySelectorAll('.tab-content');
  tabs.forEach(tab => tab.classList.remove('active'));
  document.getElementById(tabName).classList.add('active');
  
  const buttons = document.querySelectorAll('.tab-button');
  buttons.forEach(btn => btn.classList.remove('active'));
  if (event) event.target.classList.add('active');
}

/**
 * Candidate Slider Logic
 */
const numSlider = document.getElementById('numCandidates');
if (numSlider) {
  numSlider.addEventListener('input', (e) => {
    document.getElementById('sliderValue').textContent = e.target.value;
  });
}

/**
 * Single Prediction Logic (Manual Input)
 */
async function predict(){
  const skills = document.getElementById("skills").value;
  const education = document.getElementById("education").value;
  const certifications = document.getElementById("certifications").value;
  const job_role = document.getElementById("job_role").value;

  const response = await fetch("http://127.0.0.1:5000/predict",{
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body:JSON.stringify({
      skills:skills,
      education:education,
      certifications:certifications,
      job_role:job_role
    })
  });

  const data = await response.json();
  document.getElementById("result").innerHTML = "<h3>Decision: " + data.decision + "</h3>";
}

/**
 * Bulk Candidate Screening (CSV Upload)
 */
document.getElementById('candidateForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('csvFile');
  const jobDesc = document.getElementById('jobDesc').value.trim();

  if (!fileInput.files[0]) {
    document.getElementById('candidateResult').innerHTML = '<div style="color:#dc2626; padding:20px; background:#fef2f2; border-radius:8px; text-align:center;"><strong>❌ Error:</strong> Please select a CSV file.</div>';
    return;
  }

  if (!jobDesc) {
    document.getElementById('candidateResult').innerHTML = '<div style="color:#dc2626; padding:20px; background:#fef2f2; border-radius:8px; text-align:center;"><strong>❌ Error:</strong> Please enter a job description.</div>';
    return;
  }

  const formData = new FormData();
  formData.append('csv', fileInput.files[0]);
  formData.append('job_desc', jobDesc);
  formData.append('num_candidates', document.getElementById('numCandidates').value);

  document.getElementById('candidateResult').innerHTML = '<div style="text-align:center; padding:40px;"><div style="display:inline-block; width:40px; height:40px; border:4px solid #e8e0d4; border-top:4px solid #e8541a; border-radius:50%; animation:spin 1s linear infinite;"></div><p style="margin-top:20px; color:#9d958c;">Training ML models and finding best candidates...</p></div>';

  try {
    const response = await fetch('http://127.0.0.1:5000/train_and_candidates', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      document.getElementById('candidateResult').innerHTML = `<div style="color:#dc2626; padding:20px; background:#fef2f2; border-radius:8px; text-align:center;"><strong>❌ Error:</strong> ${data.error}</div>`;
      return;
    }

    // Success message with model accuracy
    let html = `
    <div style="background:#f0f9f8; border:1px solid #059669; border-radius:12px; padding:20px; margin-bottom:24px; text-align:center;">
      <div style="font-size:48px; color:#059669; margin-bottom:8px;">🎯</div>
      <h3 style="color:#059669; margin:0; font-size:18px;">Model Training Complete!</h3>
      <p style="color:#059669; margin:8px 0 0 0; font-weight:600;">Best Model Accuracy: ${data.best_accuracy.toFixed(2)}%</p>
    </div>

    <div style="background:white; border-radius:12px; padding:24px; border:1px solid #e8e0d4;">
      <h3 style="margin:0 0 20px 0; color:#1a1612;">Top ${data.candidates.length} Matching Candidates</h3>
      <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
          <thead>
            <tr style="background:#e8541a; color:white;">
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Name</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Skills</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Experience</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Education</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Job Role</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Salary</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">AI Score</th>
              <th style="padding:12px; text-align:left; border:1px solid #ddd;">Match Score</th>
            </tr>
          </thead>
          <tbody>`;

    data.candidates.forEach((c, index) => {
      const matchScore = c['Match Score'].toFixed(3);
      const matchColor = matchScore > 0.8 ? '#059669' : matchScore > 0.6 ? '#d97706' : '#dc2626';

      html += `<tr style="background:${index % 2 === 0 ? '#fdf9f4' : 'white'};">
        <td style="padding:12px; border:1px solid #ddd; font-weight:600;">${c['Name'] || 'N/A'}</td>
        <td style="padding:12px; border:1px solid #ddd;">${c['Skills'] || 'N/A'}</td>
        <td style="padding:12px; border:1px solid #ddd;">${c['Experience (Years)'] || 'N/A'} years</td>
        <td style="padding:12px; border:1px solid #ddd;">${c['Education'] || 'N/A'}</td>
        <td style="padding:12px; border:1px solid #ddd;">${c['Job Role'] || 'N/A'}</td>
        <td style="padding:12px; border:1px solid #ddd; font-weight:600;">$${c['Salary Expectation ($)'] || 'N/A'}</td>
        <td style="padding:12px; border:1px solid #ddd;">${c['AI Score (0-100)'] || 'N/A'}/100</td>
        <td style="padding:12px; border:1px solid #ddd; font-weight:bold; color:${matchColor};">${matchScore}</td>
      </tr>`;
    });

    html += `
          </tbody>
        </table>
      </div>
    </div>`;

    document.getElementById('candidateResult').innerHTML = html;

  } catch (err) {
    document.getElementById('candidateResult').innerHTML = `<div style="color:#dc2626; padding:20px; background:#fef2f2; border-radius:8px; text-align:center;"><strong>❌ Error:</strong> ${err.message}</div>`;
  }
});

// Update CSV file label when a file is chosen
const csvFileInput = document.getElementById('csvFile');
if (csvFileInput) {
  csvFileInput.addEventListener('change', function() {
    const label = document.getElementById('csvFileName');
    if (!label) return;

    if (this.files.length > 0) {
      const file = this.files[0];
      label.textContent = `${file.name} — ${Math.round(file.size / 1024)} KB`;
    } else {
      label.textContent = 'Drag and drop a file here or click Browse files';
    }
  });
}

/**
 * UI EVENT LISTENERS (File Drops & Counts)
 */

// Track file upload and update status (ATS Tab)
const resumeFileInput = document.getElementById('resumeFile');
if (resumeFileInput) {
  resumeFileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
      const fileName = this.files[0].name;
      const fileSize = (this.files[0].size / 1024 / 1024).toFixed(2);
      document.getElementById('fileNameDisplay').innerHTML = `✓ ${fileName}<br><small style="font-size:11px; color:#9d958c;">${fileSize} MB</small>`;
      document.getElementById('fileStatus').style.display = 'block';
      document.getElementById('statusBox').innerHTML = `<span style="color:#059669;">✓ ${fileName}</span>`;
    }
  });
}

// Track job description character count
const jobDescArea = document.getElementById('jobDesc2');
if (jobDescArea) {
  jobDescArea.addEventListener('input', function() {
    const charCount = this.value.length;
    const charDisplay = document.getElementById('charCount');
    if (charDisplay) charDisplay.textContent = charCount;
    
    const statusBox = document.getElementById('jobDescStatus');
    if (statusBox) {
        if (charCount > 100) {
          statusBox.innerHTML = `<span style="color:#059669;">✓ ${charCount} characters</span>`;
        } else if (charCount > 0) {
          statusBox.innerHTML = `<span style="color:#d97706;">${charCount} (add more)</span>`;
        } else {
          statusBox.innerHTML = `Awaiting input`;
        }
    }
  });
}

/**
 * ATS RESUME CHECKER LOGIC (The Core Feature)
 */
async function analyzeResume() {
  const resumeFile = document.getElementById('resumeFile');
  const jobDesc = document.getElementById('jobDesc2');
  const resultContainer = document.getElementById('atsResult');
  
  if (!resumeFile.files[0]) {
    resultContainer.innerHTML = '<p style="color:#dc2626;">Please select a resume file.</p>';
    return;
  }
  
  if (!jobDesc.value.trim()) {
    resultContainer.innerHTML = '<p style="color:#dc2626;">Please paste a job description.</p>';
    return;
  }
  
  const formData = new FormData();
  formData.append('resume', resumeFile.files[0]);
  formData.append('job_desc', jobDesc.value);
  
  resultContainer.innerHTML = '<p style="text-align:center;">Analyzing resume with AI...</p>';
  
  try {
    const response = await fetch('http://127.0.0.1:5000/analyze_resume', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.error) {
      resultContainer.innerHTML = `<p style="color:#dc2626;">Error: ${data.error}</p>`;
      return;
    }
    
    // BUILD THE RESULTS UI
    let html = `
    <div style="background:white; border-radius:16px; padding:40px; margin-bottom:32px; border:1px solid #e8e0d4; box-shadow: var(--shadow-lg); transition: all 0.3s ease;">
      <div style="display:flex; align-items:center; justify-content:center; gap:48px; flex-wrap:wrap;">
        <div style="flex:0 0 auto;">
          <div style="width:200px; height:200px; border-radius:50%; background: linear-gradient(135deg, ${data.level_color}10 0%, ${data.level_color}25 100%); border: 8px solid ${data.level_color}20; display:flex; align-items:center; justify-content:center; position: relative;">
            <div style="position: absolute; width: 100%; height: 100%; border-radius: 50%; border: 4px solid ${data.level_color}; border-top-color: transparent; border-right-color: transparent; transform: rotate(45deg);"></div>
            <div style="text-align:center;">
              <div style="font-size:54px; font-weight:900; color:${data.level_color}; line-height:1;">${data.match_percent}%</div>
              <div style="font-size:11px; color:${data.level_color}; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-top:8px;">Match Score</div>
            </div>
          </div>
        </div>
        <div style="flex:1; min-width:300px;">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <h3 style="font-size:28px; font-weight:800; color:${data.level_color}; margin:0;">${data.match_level}</h3>
            <span style="background:${data.level_color}20; color:${data.level_color}; padding:4px 12px; border-radius:99px; font-size:12px; font-weight:700;">AI AUDIT</span>
          </div>
          <p style="color:#9d958c; font-size:14px; margin:0 0 20px 0;">We found <strong>${data.total_matched}</strong> technical matches out of <strong>${data.total_required}</strong> required skills identified in the job description.</p>
          <div style="background:#fdf9f4; border-left:4px solid ${data.level_color}; padding:20px; border-radius:12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);">
            <p style="color:#4a4540; line-height:1.6; font-size:15px; margin:0;">${data.recommendation}</p>
          </div>
        </div>
      </div>
    </div>
    `;
    
    // Section: Matched Skills
    if (data.matched_skills && data.matched_skills.length > 0) {
      html += `
      <div style="background:white; border-radius:12px; padding:25px; margin-bottom:20px; border:1px solid #e8e0d4;">
        <h4 style="font-size:15px; font-weight:600; color:#059669; margin-bottom:15px;">✓ Key Skills Found</h4>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
          ${data.matched_skills.map(skill => `<span style="background:#e8f5f0; color:#059669; padding:6px 12px; border-radius:20px; font-size:12px; border:1px solid #05966930;">${skill}</span>`).join('')}
        </div>
      </div>`;
    }
    
    // Section: Missing Gaps
    if (data.total_missing > 0) {
      html += `
      <div style="background:white; border-radius:12px; padding:25px; border:1px solid #e8e0d4; margin-bottom:24px;">
        <h4 style="font-size:15px; font-weight:600; color:#dc2626; margin-bottom:15px;">⚠ Strategic Gaps to Fix</h4>`;
      
      const cats = { "Technical": data.missing_categories.technical, "Tools": data.missing_categories.tools, "Soft Skills": data.missing_categories.soft_skills, "Other": data.missing_categories.other };
      
      for (const [name, list] of Object.entries(cats)) {
        if (list && list.length > 0) {
          html += `<p style="font-size:11px; font-weight:700; color:#9d958c; text-transform:uppercase; margin:15px 0 8px 0;">${name}</p>
                   <div style="display:flex; flex-wrap:wrap; gap:8px;">
                   ${list.map(s => `<span style="background:#fff1f0; color:#dc2626; padding:6px 12px; border-radius:20px; font-size:12px; border:1px solid #dc262620;">${s}</span>`).join('')}
                   </div>`;
        }
      }
      html += `</div>`;
    }

    if (data.improvement_tips && data.improvement_tips.length > 0) {
      html += `
      <div style="background:white; border-radius:12px; padding:25px; border:1px solid #e8e0d4; margin-bottom:24px;">
        <h4 style="font-size:15px; font-weight:600; color:#2563d4; margin-bottom:15px;">🛠 How to Improve</h4>
        <ul style="padding-left:20px; color:#4a4540; font-size:14px; line-height:1.8; margin:0;">
          ${data.improvement_tips.map(tip => `<li style="margin-bottom:10px;">${tip}</li>`).join('')}
        </ul>
      </div>`;
    }

    if (data.learning_resources && data.learning_resources.length > 0) {
      html += `
      <div style="background:white; border-radius:12px; padding:25px; border:1px solid #e8e0d4; margin-bottom:24px;">
        <h4 style="font-size:15px; font-weight:600; color:#059669; margin-bottom:15px;">📚 Where to Learn Missing Skills</h4>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px;">
          ${data.learning_resources.map(res => `
            <a href="${res.url}" target="_blank" style="text-decoration:none; color:inherit; display:block;">
              <div style="background:#f0f9f8; border-radius:12px; padding:16px; border:1px solid #d1fae5; transition: all 0.2s ease; cursor:pointer;" onmouseover="this.style.borderColor='#059669'; this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='#d1fae5'; this.style.transform='translateY(0)'">
                <div style="font-weight:700; color:#065f46; margin-bottom:4px; display:flex; align-items:center; gap:8px;">
                  <span>${res.platform}</span>
                  <span style="font-size:10px; background:#d1fae5; padding:2px 6px; border-radius:4px; font-weight:600;">VISIT ↗</span>
                </div>
                <div style="font-size:12px; color:#064e3b; line-height:1.4;">${res.description}</div>
              </div>
            </a>
          `).join('')}
        </div>
      </div>`;
    }
    
    resultContainer.innerHTML = html;
  } catch (err) {
    resultContainer.innerHTML = `<p style="color:#dc2626;">System Error: ${err.message}</p>`;
  }
}