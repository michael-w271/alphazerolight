document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Add fade-in class to elements we want to animate on scroll
    const animatedElements = document.querySelectorAll('.feature-card, .training-card, .section-title');
    animatedElements.forEach(el => {
        el.classList.add('fade-in');
        el.style.opacity = '0'; // Start hidden
        observer.observe(el);
    });

    // Fix for elements that are already fade-in from CSS (hero elements)
    // We don't need to observe them, they animate on load.
});

// Add visible class style dynamically if not in CSS
const style = document.createElement('style');
style.innerHTML = `
    .visible {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);
