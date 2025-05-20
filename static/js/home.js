let lastScrollY = window.scrollY;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
  const currentScroll = window.scrollY;

  if (currentScroll === 0) {
    // Back at the top — make navbar transparent again
    navbar.classList.remove('navbar-visible');
    navbar.style.top = '0';
  } else if (currentScroll < lastScrollY) {
    // Scrolling up — show navbar with background
    navbar.style.top = '0';
    navbar.classList.add('navbar-visible');
  } else {
    // Scrolling down — hide navbar
    navbar.style.top = '-100px';
    navbar.classList.remove('navbar-visible');
  }

  lastScrollY = currentScroll;
});
