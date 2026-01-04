const yearEl = document.getElementById('year');
if (yearEl) {
    yearEl.textContent = new Date().getFullYear();
}

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            entry.target.classList.add('active');
        }
    });
}, { threshold: 0.15 });

const revealTargets = document.querySelectorAll('[data-reveal]');
revealTargets.forEach((el) => {
    el.classList.add('reveal');
    revealObserver.observe(el);
});

const mapEl = document.getElementById('propagation-map');
if (mapEl) {
    const mapObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                mapEl.classList.add('active');
            }
        });
    }, { threshold: 0.25 });

    mapObserver.observe(mapEl);
}

const terminalNodes = document.querySelectorAll('[data-terminal-text]');
terminalNodes.forEach((el) => {
    const text = el.dataset.terminalText || '';
    if (!text) {
        return;
    }

    let index = 0;

    const typeNext = () => {
        if (index <= text.length) {
            el.textContent = text.slice(0, index);
            index += 1;
            setTimeout(typeNext, 35 + Math.random() * 55);
        } else {
            setTimeout(() => {
                index = 0;
                el.textContent = '';
                typeNext();
            }, 2200);
        }
    };

    typeNext();
});

const magneticButtons = document.querySelectorAll('.magnetic');
magneticButtons.forEach((btn) => {
    btn.addEventListener('mousemove', (event) => {
        const rect = btn.getBoundingClientRect();
        const x = event.clientX - rect.left - rect.width / 2;
        const y = event.clientY - rect.top - rect.height / 2;
        btn.style.transform = `translate(${x * 0.1}px, ${y * 0.2}px)`;
    });

    btn.addEventListener('mouseleave', () => {
        btn.style.transform = 'translate(0, 0)';
    });
});

const staggerGroups = document.querySelectorAll('[data-stagger]');
staggerGroups.forEach((group) => {
    const delay = Number(group.dataset.stagger) || 150;
    group.querySelectorAll('[data-stagger-item]').forEach((item, index) => {
        item.style.transitionDelay = `${index * delay}ms`;
    });
});

const pricingState = {
    billing: 'monthly',
};

const billingButtons = document.querySelectorAll('[data-billing-option]');
const proPriceEl = document.querySelector('[data-pro-price]');
const billingNoteEl = document.querySelector('[data-billing-note]');
const estimatorEl = document.querySelector('[data-usage-estimator]');

const tracesValueEl = estimatorEl ? estimatorEl.querySelector('[data-traces-value]') : null;
const sliderEl = estimatorEl ? estimatorEl.querySelector('[data-usage-slider]') : null;
const estimateValueEl = estimatorEl ? estimatorEl.querySelector('[data-estimate-value]') : null;
const estimateLabelEl = estimatorEl ? estimatorEl.querySelector('[data-estimate-label]') : null;

function getBasePrice(mode) {
    return mode === 'annual' ? 219 : 249;
}

function computeUsageCost(tps, mode) {
    const base = getBasePrice(mode);
    const extraBlocks = Math.max(tps - 250, 0) / 100;
    const perBlock = mode === 'annual' ? 15 : 18;
    return Math.round(base + extraBlocks * perBlock);
}

function updatePricingUI() {
    if (proPriceEl) {
        proPriceEl.textContent = `$${getBasePrice(pricingState.billing)}`;
    }
    if (billingNoteEl) {
        billingNoteEl.textContent = pricingState.billing === 'annual'
            ? '/month (billed annually, save 12%)'
            : '/month';
    }
    if (estimateLabelEl) {
        estimateLabelEl.textContent = pricingState.billing === 'annual'
            ? '/month (annual plan)'
            : '/month';
    }
    if (estimatorEl && sliderEl && tracesValueEl && estimateValueEl) {
        const tps = Number(sliderEl.value);
        tracesValueEl.textContent = tps.toLocaleString();
        const estCost = computeUsageCost(tps, pricingState.billing);
        estimateValueEl.textContent = `$${estCost.toLocaleString()}`;
    }
}

if (billingButtons.length) {
    billingButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.billingOption;
            if (!mode || pricingState.billing === mode) {
                return;
            }
            pricingState.billing = mode;
            billingButtons.forEach((button) => button.classList.toggle('active', button === btn));
            updatePricingUI();
        });
    });
}

if (sliderEl) {
    sliderEl.addEventListener('input', () => {
        updatePricingUI();
    });
    updatePricingUI();
} else {
    updatePricingUI();
}
