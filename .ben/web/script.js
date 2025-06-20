// Configuration - Auto-detect API URL based on current host
const getApiBaseUrl = () => {
    const currentHost = window.location.hostname;
    if (currentHost === 'localhost' || currentHost === '127.0.0.1') {
        return 'http://localhost:8000';
    } else {
        return `http://${currentHost}:8000`;
    }
};

const API_BASE_URL = getApiBaseUrl();
const DEFAULT_RESULTS = 10;
const SEARCH_DELAY = 300; // Debounce delay in ms

// DOM Elements
let searchInput, searchButton, resultsCount, includeText;
let loading, resultsSection, resultsHeader, resultsContainer;
let statusDot, statusText, statsGrid, refreshStats;
let errorModal, modalBody, closeModal;

// State
let searchTimeout;
let currentQuery = '';
let isSearching = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    setupEventListeners();
    checkSystemHealth();
    loadSystemStats();
});

function initializeElements() {
    // Search elements
    searchInput = document.getElementById('searchInput');
    searchButton = document.getElementById('searchButton');
    resultsCount = document.getElementById('resultsCount');
    includeText = document.getElementById('includeText');
    
    // Display elements
    loading = document.getElementById('loading');
    resultsSection = document.getElementById('resultsSection');
    resultsHeader = document.getElementById('resultsHeader');
    resultsContainer = document.getElementById('resultsContainer');
    
    // Status elements
    statusDot = document.getElementById('statusDot');
    statusText = document.getElementById('statusText');
    
    // Stats elements
    statsGrid = document.getElementById('statsGrid');
    refreshStats = document.getElementById('refreshStats');
    
    // Modal elements
    errorModal = document.getElementById('errorModal');
    modalBody = document.getElementById('modalBody');
    closeModal = document.getElementById('closeModal');
}

function setupEventListeners() {
    // Search functionality
    searchButton.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Real-time search (debounced)
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = searchInput.value.trim();
        
        if (query.length > 2) {
            searchTimeout = setTimeout(() => {
                if (query !== currentQuery) {
                    handleSearch();
                }
            }, SEARCH_DELAY);
        }
    });
    
    // Stats refresh
    refreshStats.addEventListener('click', loadSystemStats);
    
    // Modal close
    closeModal.addEventListener('click', hideModal);
    errorModal.addEventListener('click', function(e) {
        if (e.target === errorModal) {
            hideModal();
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            hideModal();
        }
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            searchInput.focus();
        }
    });
}

async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const health = await response.json();
        
        updateStatusIndicator(health);
        
        if (!health.redis_connected || !health.sqlite_connected || !health.model_loaded) {
            console.warn('System health issues detected:', health);
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator({
            status: 'offline',
            redis_connected: false,
            sqlite_connected: false,
            model_loaded: false
        });
    }
}

function updateStatusIndicator(health) {
    const isHealthy = health.status === 'healthy';
    
    statusDot.className = `status-dot ${isHealthy ? 'online' : 'offline'}`;
    statusText.textContent = isHealthy ? 'System Online' : 'System Issues';
    
    if (!isHealthy) {
        const issues = [];
        if (!health.redis_connected) issues.push('Redis');
        if (!health.sqlite_connected) issues.push('SQLite');
        if (!health.model_loaded) issues.push('Model');
        
        statusText.textContent = `Issues: ${issues.join(', ')}`;
    }
}

async function loadSystemStats() {
    try {
        refreshStats.style.transform = 'rotate(180deg)';
        
        const response = await fetch(`${API_BASE_URL}/stats`);
        const stats = await response.json();
        
        displaySystemStats(stats);
    } catch (error) {
        console.error('Failed to load stats:', error);
        showError('Failed to load system statistics');
    } finally {
        setTimeout(() => {
            refreshStats.style.transform = 'rotate(0deg)';
        }, 300);
    }
}

function displaySystemStats(stats) {
    const statsItems = [
        {
            label: 'Redis Documents',
            value: formatNumber(stats.redis_documents),
            color: '#667eea'
        },
        {
            label: 'SQLite Documents',
            value: formatNumber(stats.sqlite_documents),
            color: '#27ae60'
        },
        {
            label: 'Coverage Ratio',
            value: `${(stats.coverage_ratio * 100).toFixed(1)}%`,
            color: '#f39c12'
        },
        {
            label: 'Redis Memory',
            value: stats.redis_memory,
            color: '#e74c3c'
        }
    ];
    
    statsGrid.innerHTML = statsItems.map(item => `
        <div class="stat-item">
            <div class="stat-value" style="color: ${item.color}">${item.value}</div>
            <div class="stat-label">${item.label}</div>
        </div>
    `).join('');
}

async function handleSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }
    
    if (isSearching) {
        return; // Prevent multiple simultaneous searches
    }
    
    currentQuery = query;
    isSearching = true;
    
    showLoading();
    hideResults();
    disableSearch();
    
    try {
        const results = await performSearch(query);
        displayResults(results);
    } catch (error) {
        console.error('Search failed:', error);
        showError(`Search failed: ${error.message}`);
    } finally {
        hideLoading();
        enableSearch();
        isSearching = false;
    }
}

async function performSearch(query) {
    const searchParams = {
        query: query,
        k: parseInt(resultsCount.value),
        include_text: includeText.checked
    };
    
    const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(searchParams)
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    return await response.json();
}

function displayResults(data) {
    // Update results header
    resultsHeader.innerHTML = `
        <div class="results-summary">
            <div class="results-info">
                <h2>Search Results</h2>
                <div class="results-meta">
                    Found ${data.total_results} result${data.total_results !== 1 ? 's' : ''} for "${escapeHtml(data.query)}"
                </div>
            </div>
            <div class="search-time">
                ${data.search_time_ms.toFixed(1)}ms
            </div>
        </div>
    `;
    
    // Display results
    if (data.results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <h3>No results found</h3>
                <p>Try different keywords or check your spelling.</p>
            </div>
        `;
    } else {
        resultsContainer.innerHTML = data.results.map((result, index) => 
            createResultItem(result, index)
        ).join('');
        
        // Add expand/collapse functionality
        setupResultInteractions();
    }
    
    showResults();
}

function createResultItem(result, index) {
    const similarity = (result.similarity * 100).toFixed(1);
    const preview = result.doc_text.length > 300;
    const displayText = preview ? result.doc_text.substring(0, 300) + '...' : result.doc_text;
    
    return `
        <div class="result-item" data-index="${index}">
            <div class="result-header">
                <div class="result-id">${escapeHtml(result.doc_id)}</div>
                <div class="similarity-score">${similarity}%</div>
            </div>
            
            <div class="result-content ${preview ? 'preview' : ''}" data-full-text="${escapeHtml(result.doc_text)}">
                ${escapeHtml(displayText)}
            </div>
            
            <div class="result-meta">
                <span>Length: ${formatNumber(result.doc_length)} chars</span>
                <span>Source: ${escapeHtml(result.source)}</span>
                ${preview ? '<button class="expand-btn" data-action="expand">Show More</button>' : ''}
            </div>
        </div>
    `;
}

function setupResultInteractions() {
    document.querySelectorAll('.expand-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const resultItem = this.closest('.result-item');
            const contentDiv = resultItem.querySelector('.result-content');
            const isExpanded = this.dataset.action === 'collapse';
            
            if (isExpanded) {
                // Collapse
                const preview = contentDiv.dataset.fullText.substring(0, 300) + '...';
                contentDiv.innerHTML = escapeHtml(preview);
                contentDiv.classList.add('preview');
                this.textContent = 'Show More';
                this.dataset.action = 'expand';
            } else {
                // Expand
                contentDiv.innerHTML = escapeHtml(contentDiv.dataset.fullText);
                contentDiv.classList.remove('preview');
                this.textContent = 'Show Less';
                this.dataset.action = 'collapse';
            }
        });
    });
    
    // Add click-to-copy functionality for document IDs
    document.querySelectorAll('.result-id').forEach(idElement => {
        idElement.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent).then(() => {
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.style.color = '#27ae60';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.color = '#667eea';
                }, 1000);
            }).catch(() => {
                console.warn('Failed to copy to clipboard');
            });
        });
        
        idElement.title = 'Click to copy document ID';
        idElement.style.cursor = 'pointer';
    });
}

function showLoading() {
    loading.classList.add('show');
}

function hideLoading() {
    loading.classList.remove('show');
}

function showResults() {
    resultsSection.classList.add('show');
}

function hideResults() {
    resultsSection.classList.remove('show');
}

function disableSearch() {
    searchButton.disabled = true;
    searchButton.innerHTML = `
        <div class="spinner" style="width: 16px; height: 16px; margin: 0; border-width: 2px;"></div>
        Searching...
    `;
}

function enableSearch() {
    searchButton.disabled = false;
    searchButton.innerHTML = `
        <span class="search-icon">🔍</span>
        Search
    `;
}

function showError(message) {
    modalBody.innerHTML = `
        <p><strong>An error occurred:</strong></p>
        <p>${escapeHtml(message)}</p>
        <p>Please check the console for more details or try again.</p>
    `;
    errorModal.classList.add('show');
}

function hideModal() {
    errorModal.classList.remove('show');
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Auto-refresh health status periodically
setInterval(checkSystemHealth, 30000); // Every 30 seconds

// Make getApiBaseUrl available globally for HTML onclick handlers
window.getApiBaseUrl = getApiBaseUrl;

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(() => {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Page loaded in ${loadTime}ms`);
        }, 0);
    });
}

// Service worker registration (for future PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment when service worker is implemented
        // navigator.serviceWorker.register('/sw.js');
    });
} 