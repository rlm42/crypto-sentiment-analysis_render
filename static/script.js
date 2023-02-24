document.getElementById("news-tab").addEventListener("click", function(event) {
    event.preventDefault();
    document.getElementById("tweets").classList.remove("show", "active");
    document.getElementById("tweets-tab").classList.remove("active");
    document.getElementById("sentiment").classList.remove("show", "active");
    document.getElementById("sentiment-tab").classList.remove("active");
    document.getElementById("news").classList.add("show", "active");
    document.getElementById("news-tab").classList.add("active");
  });
  
  
  document.getElementById("tweets-tab").addEventListener("click", function(event) {
    event.preventDefault();
    document.getElementById("news").classList.remove("show", "active");
    document.getElementById("news-tab").classList.remove("active");
    document.getElementById("sentiment").classList.remove("show", "active");
    document.getElementById("sentiment-tab").classList.remove("active");
    document.getElementById("tweets").classList.add("show", "active");
    document.getElementById("tweets-tab").classList.add("active");
  });
  
  document.getElementById("sentiment-tab").addEventListener("click", function(event) {
    event.preventDefault();
    document.getElementById("tweets").classList.remove("show", "active");
    document.getElementById("tweets-tab").classList.remove("active");
    document.getElementById("news").classList.remove("show", "active");
    document.getElementById("news-tab").classList.remove("active");
    document.getElementById("sentiment").classList.add("show", "active");
    document.getElementById("sentiment-tab").classList.add("active");
  });
  
  
  
  
      document.getElementById("backButton").addEventListener("click", function() {
        window.history.back();
      });
  
      const sentimentFilter = document.getElementById("sentimentFilter");
  sentimentFilter.addEventListener("change", function() {
    let filterValue = sentimentFilter.value.toLowerCase();
    let tweetsRows = document.querySelectorAll("#tweetsTableBody tr");
    for (const row of tweetsRows) {
      if (filterValue === "all") {
        row.style.display = "";
      } else if (row.cells[2].textContent.toLowerCase() !== filterValue) {
        row.style.display = "none";
      } else {
        row.style.display = "";
      }
    }
  });
  
  const sentimentFilterNews = document.getElementById("sentimentFilterNews");
  sentimentFilterNews.addEventListener("change", function() {
    let filterValue = sentimentFilterNews.value.toLowerCase();
    let newsRows = document.querySelectorAll("#newsTableBody tr");
    for (const row of newsRows) {
      if (filterValue === "all") {
        row.style.display = "";
      } else if (row.cells[8].textContent.toLowerCase() !== filterValue) {
        row.style.display = "none";
      } else {
        row.style.display = "";
      }
    }
      });