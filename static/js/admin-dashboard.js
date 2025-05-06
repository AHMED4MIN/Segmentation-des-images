document.addEventListener('DOMContentLoaded', function() {
    // Sidebar navigation
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            
            // Remove active classes
            document.querySelectorAll('.nav-link.active').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.dashboard-section.active').forEach(el => el.classList.remove('active'));
            
            // Add active classes
            this.classList.add('active');
            document.getElementById(targetId).classList.add('active');
        });
    });
});