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

    // Material Design 3 inspired interactions
    // Mobile navigation menu toggle
    const sidebar = document.querySelector('.site-sidebar');
    
    if (mobileMenuButton && sidebar) {
        mobileMenuButton.addEventListener('click', function() {
            sidebar.classList.toggle('open');
            const isOpen = sidebar.classList.contains('open');
            this.setAttribute('aria-expanded', isOpen);
            this.innerHTML = isOpen 
                ? '<span class="material-icons">close</span>' 
                : '<span class="material-icons">menu</span>';
        });

        // Close the menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!sidebar.contains(event.target) && !mobileMenuButton.contains(event.target)) {
                if (sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                    mobileMenuButton.setAttribute('aria-expanded', 'false');
                    mobileMenuButton.innerHTML = '<span class="material-icons">menu</span>';
                }
            }
        });
    }

    // Add ripple effect to buttons (Material Design interaction)
    function createRipple(event) {
        const button = event.currentTarget;
        
        const circle = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;
        
        circle.style.width = circle.style.height = `${diameter}px`;
        circle.style.left = `${event.clientX - button.getBoundingClientRect().left - radius}px`;
        circle.style.top = `${event.clientY - button.getBoundingClientRect().top - radius}px`;
        circle.classList.add('ripple');
        
        const ripple = button.getElementsByClassName('ripple')[0];
        if (ripple) {
            ripple.remove();
        }
        
        button.appendChild(circle);
    }
    
    // Add ripple to all buttons
    const buttons = document.querySelectorAll('.md-button, button:not(.mobile-menu-button)');
    buttons.forEach(button => {
        button.addEventListener('mousedown', createRipple);
    });

    // Add ripple to navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('mousedown', createRipple);
    });

    // Add code copy functionality
    document.querySelectorAll('pre').forEach(block => {
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        
        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = '<span class="material-icons">content_copy</span>';
        wrapper.appendChild(button);
        
        // Add click handler
        button.addEventListener('click', async () => {
            try {
                const code = block.textContent;
                await navigator.clipboard.writeText(code);
                
                // Show success
                button.innerHTML = '<span class="material-icons">check</span>';
                button.classList.add('copied');
                
                // Reset after 2 seconds
                setTimeout(() => {
                    button.innerHTML = '<span class="material-icons">content_copy</span>';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code', err);
            }
        });
    });

    // Highlight currently active section in TOC
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
            rootMargin: '-100px 0px -70%'
        });

        headings.forEach(heading => observer.observe(heading));
    }

    // Simple search implementation (can be enhanced with tools like lunr.js)
    const searchResults = document.getElementById('search-results');
    
    if (searchInput && searchResults) {
        searchInput.addEventListener('input', function() {
            const query = this.value.trim().toLowerCase();
            
            if (query.length < 2) {
                searchResults.classList.add('hidden');
                return;
            }
            
            // This is a placeholder - in a real implementation you'd
            // use a search index or API to search content
            searchResults.innerHTML = `<p>Search for "${query}" is not yet implemented.</p>`;
            searchResults.classList.remove('hidden');
        });

        // Hide search results when clicking outside
        document.addEventListener('click', function(event) {
            if (!searchInput.contains(event.target) && !searchResults.contains(event.target)) {
                searchResults.classList.add('hidden');
            }
        });
    }

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href').slice(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                e.preventDefault();
                
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
                
                // Update URL without scrolling
                history.pushState(null, null, this.getAttribute('href'));
            }
        });
    });

    // Add keyboard navigation for accessibility
    document.addEventListener('keydown', function(e) {
        // ESC key closes mobile menu
        if (e.key === 'Escape' && sidebar && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
            mobileMenuButton.setAttribute('aria-expanded', 'false');
            mobileMenuButton.innerHTML = '<span class="material-icons">menu</span>';
        }
    });
});