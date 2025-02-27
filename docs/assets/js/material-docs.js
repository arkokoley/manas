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
});