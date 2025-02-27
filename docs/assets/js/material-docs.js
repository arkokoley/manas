// Material Design 3 inspired interactions for Just the Docs
document.addEventListener('DOMContentLoaded', function() {
  // Add ripple effect to buttons (Material Design interaction)
  function createRipple(event) {
    const button = event.currentTarget;
    
    const circle = document.createElement("span");
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    const radius = diameter / 2;
    
    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${event.clientX - button.getBoundingClientRect().left - radius}px`;
    circle.style.top = `${event.clientY - button.getBoundingClientRect().top - radius}px`;
    circle.classList.add("ripple");
    
    const ripple = button.getElementsByClassName("ripple")[0];
    if (ripple) {
      ripple.remove();
    }
    
    button.appendChild(circle);
  }
  
  // Apply ripple effect to buttons
  const buttons = document.getElementsByTagName("button");
  for (const button of buttons) {
    button.addEventListener("click", createRipple);
  }
  
  // Add ripple to navigation items
  const navLinks = document.querySelectorAll('.nav-list-link');
  navLinks.forEach(link => {
    link.addEventListener('click', createRipple);
  });
  
  // Code block copy functionality
  document.querySelectorAll('pre.highlight').forEach((codeBlock) => {
    // Create wrapper for position relative
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';
    codeBlock.parentNode.insertBefore(wrapper, codeBlock);
    wrapper.appendChild(codeBlock);
    
    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'md-copy-button';
    copyButton.setAttribute('aria-label', 'Copy code');
    copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM240-120q-33 0-56.5-23.5T160-200v-560h80v560h400v80H240Zm120-200v-480 480Z"/></svg>';
    wrapper.appendChild(copyButton);
    
    // Add copy functionality
    copyButton.addEventListener('click', () => {
      const code = codeBlock.querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(() => {
        // Show success state
        copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>';
        copyButton.classList.add('copied');
        
        // Reset after delay
        setTimeout(() => {
          copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM240-120q-33 0-56.5-23.5T160-200v-560h80v560h400v80H240Zm120-200v-480 480Z"/></svg>';
          copyButton.classList.remove('copied');
        }, 2000);
      });
    });
  });
  
  // Add Material Design elevation to cards
  document.querySelectorAll('.card').forEach(card => {
    card.classList.add('md-elevation-1');
    card.addEventListener('mouseenter', () => {
      card.classList.replace('md-elevation-1', 'md-elevation-2');
    });
    card.addEventListener('mouseleave', () => {
      card.classList.replace('md-elevation-2', 'md-elevation-1');
    });
  });
  
  // Improve anchor link scrolling with offset for fixed header
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').slice(1);
      if (!targetId) return;
      
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        const headerOffset = 80;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
        
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
        
        // Update URL without jumping
        history.pushState(null, null, `#${targetId}`);
      }
    });
  });
  
  // Handle mobile menu toggle for Just the Docs
  const menuButton = document.querySelector('.js-main-nav-trigger');
  if (menuButton) {
    const siteNav = document.querySelector('.site-nav');
    menuButton.addEventListener('click', () => {
      setTimeout(() => {
        if (siteNav.classList.contains('nav-open')) {
          menuButton.classList.add('nav-open');
        } else {
          menuButton.classList.remove('nav-open');
        }
      }, 50);
    });
  }
  
  // Theme handling with transition smoothing
  const TRANSITION_DURATION = 300;
  const root = document.documentElement;
  const themeToggle = document.querySelector('.theme-toggle');
  
  const setTheme = (theme) => {
    root.classList.add('theme-transitioning');
    root.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
    
    setTimeout(() => {
      root.classList.remove('theme-transitioning');
    }, TRANSITION_DURATION);
  };

  // Theme toggle click handler
  themeToggle?.addEventListener('click', () => {
    const newTheme = root.classList.contains('dark') ? 'light' : 'dark';
    setTheme(newTheme);
  });

  // System theme change handler
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
      setTheme(e.matches ? 'dark' : 'light');
    }
  });

  // Code block enhancements
  document.querySelectorAll('pre.highlight').forEach(block => {
    // Create wrapper if not exists
    let wrapper = block.closest('.code-block-wrapper');
    if (!wrapper) {
      wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);
    }

    // Add copy button if not exists
    if (!wrapper.querySelector('.copy-button')) {
      const button = document.createElement('button');
      button.className = 'md-button-icon copy-button';
      button.innerHTML = '<span class="material-icons">content_copy</span>';
      button.setAttribute('aria-label', 'Copy code');
      wrapper.appendChild(button);

      // Copy functionality
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(block.textContent);
          button.innerHTML = '<span class="material-icons">check</span>';
          button.classList.add('copied');
          
          setTimeout(() => {
            button.innerHTML = '<span class="material-icons">content_copy</span>';
            button.classList.remove('copied');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy code:', err);
        }
      });
    }
  });

  // Add ripple effect to buttons
  document.addEventListener('click', (e) => {
    const target = e.target.closest('.md-button, .md-button-icon, .nav-link');
    if (!target) return;

    const ripple = document.createElement('span');
    ripple.className = 'ripple';
    target.appendChild(ripple);

    const rect = target.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size/2;
    const y = e.clientY - rect.top - size/2;

    ripple.style.width = ripple.style.height = `${size}px`;
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;

    ripple.addEventListener('animationend', () => ripple.remove());
  });

  // Mobile menu handling
  const mobileMenuButton = document.querySelector('.mobile-menu-button');
  const sidebar = document.querySelector('.site-sidebar');
  const scrim = document.querySelector('.md-scrim');

  const toggleMobileMenu = (show) => {
    sidebar?.classList.toggle('show', show);
    scrim?.classList.toggle('show', show);
    document.body.style.overflow = show ? 'hidden' : '';
    mobileMenuButton?.setAttribute('aria-expanded', show);
  };

  mobileMenuButton?.addEventListener('click', () => {
    const isExpanded = mobileMenuButton.getAttribute('aria-expanded') === 'true';
    toggleMobileMenu(!isExpanded);
  });

  scrim?.addEventListener('click', () => toggleMobileMenu(false));

  // TOC highlighting
  const tocLinks = document.querySelectorAll('.table-of-contents a');
  const headings = Array.from(document.querySelectorAll('h2[id], h3[id]'));

  if (headings.length && tocLinks.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            tocLinks.forEach(link => {
              link.classList.toggle(
                'active',
                link.getAttribute('href') === `#${entry.target.id}`
              );
            });
          }
        });
      },
      { rootMargin: '-100px 0px -70% 0px' }
    );

    headings.forEach(heading => observer.observe(heading));
  }

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const targetId = anchor.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      
      if (target) {
        e.preventDefault();
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        history.pushState(null, null, `#${targetId}`);
      }
    });
  });

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    // Close mobile menu on ESC
    if (e.key === 'Escape' && sidebar?.classList.contains('show')) {
      toggleMobileMenu(false);
    }
  });
});