document.addEventListener('DOMContentLoaded', function() {
    // Add active class to current nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (currentPath.includes(link.getAttribute('href'))) {
            link.classList.add('active');
        }
    });

    // Handle mobile menu toggle
    const mobileMenuButton = document.querySelector('.mobile-menu-button');
    const siteNav = document.querySelector('.site-nav');
    
    if (mobileMenuButton && siteNav) {
        mobileMenuButton.addEventListener('click', () => {
            siteNav.classList.toggle('show');
            mobileMenuButton.setAttribute(
                'aria-expanded',
                siteNav.classList.contains('show')
            );
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!siteNav.contains(e.target) && !mobileMenuButton.contains(e.target)) {
                siteNav.classList.remove('show');
                mobileMenuButton.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // Copy code button functionality
    document.querySelectorAll('pre.highlight').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        
        block.appendChild(button);
        
        button.addEventListener('click', async () => {
            const code = block.querySelector('code').textContent;
            try {
                await navigator.clipboard.writeText(code);
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                button.textContent = 'Failed';
            }
        });
    });

    // Table of contents highlighting
    const headings = document.querySelectorAll('h2[id], h3[id]');
    const tocLinks = document.querySelectorAll('.toc a');
    
    if (headings.length && tocLinks.length) {
        const observerCallback = (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    tocLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === '#' + entry.target.id) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        };

        const observer = new IntersectionObserver(observerCallback, {
            rootMargin: '-100px 0px -66%'
        });

        headings.forEach(heading => observer.observe(heading));
    }

    // Search functionality
    const searchInput = document.getElementById('search-input');
    
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            // Implement search functionality here
            // This would typically integrate with a search service
            console.log('Search query:', query);
        });

        // Handle search form submission
        searchInput.closest('form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            const query = searchInput.value.toLowerCase();
            // Implement search submission
            console.log('Search submitted:', query);
        });
    }

    // Active link highlighting
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = anchor.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Dark mode toggle
    const darkModeToggle = document.querySelector('.dark-mode-toggle');
    if (!darkModeToggle) return;

    const updateDarkMode = (isDark) => {
        document.documentElement.classList.toggle('dark', isDark);
        localStorage.setItem('darkMode', isDark ? 'true' : 'false');
    };

    // Check for saved dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    updateDarkMode(savedDarkMode);

    darkModeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        updateDarkMode(isDark);
    });
});