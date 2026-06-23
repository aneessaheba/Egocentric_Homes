/*
 * Local-only account system for HumanLoop.
 *
 * No backend exists yet, so accounts and sessions are stored in this
 * browser's localStorage — passwords included, in plain text. This is
 * fine for local testing on one machine, but it is NOT secure and must
 * be replaced with a real backend (hashed passwords, server-side
 * sessions) before this site is ever made public.
 */

const HL_USERS_KEY = 'hl_users';
const HL_SESSION_KEY = 'hl_session';

function hlGetUsers() {
    try {
        return JSON.parse(localStorage.getItem(HL_USERS_KEY)) || [];
    } catch {
        return [];
    }
}

function hlSaveUsers(users) {
    localStorage.setItem(HL_USERS_KEY, JSON.stringify(users));
}

function hlGetSession() {
    const email = localStorage.getItem(HL_SESSION_KEY);
    if (!email) return null;
    return hlGetUsers().find(u => u.email === email) || null;
}

function hlRegister(event) {
    event.preventDefault();
    const name = document.getElementById('register-name').value.trim();
    const email = document.getElementById('register-email').value.trim().toLowerCase();
    const password = document.getElementById('register-password').value;
    const errorEl = document.getElementById('register-error');
    errorEl.classList.add('hidden');

    if (!name || !email || !password) {
        errorEl.textContent = 'Please fill in every field.';
        errorEl.classList.remove('hidden');
        return;
    }

    const users = hlGetUsers();
    if (users.some(u => u.email === email)) {
        errorEl.textContent = 'An account with that email already exists.';
        errorEl.classList.remove('hidden');
        return;
    }

    users.push({ name, email, password });
    hlSaveUsers(users);
    localStorage.setItem(HL_SESSION_KEY, email);
    window.location.href = 'index.html';
}

function hlSignIn(event) {
    event.preventDefault();
    const email = document.getElementById('signin-email').value.trim().toLowerCase();
    const password = document.getElementById('signin-password').value;
    const errorEl = document.getElementById('signin-error');
    errorEl.classList.add('hidden');

    const user = hlGetUsers().find(u => u.email === email && u.password === password);
    if (!user) {
        errorEl.textContent = 'No account matches that email and password.';
        errorEl.classList.remove('hidden');
        return;
    }

    localStorage.setItem(HL_SESSION_KEY, email);
    window.location.href = 'index.html';
}

function hlLogout() {
    localStorage.removeItem(HL_SESSION_KEY);
    window.location.href = 'index.html';
}

function hlEscapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Swap the nav's Sign In / Register links for a logged-in state, on every page that includes this script.
function hlRenderNavAuthState() {
    const container = document.getElementById('nav-auth');
    if (!container) return;
    const session = hlGetSession();
    if (!session) return;

    container.innerHTML = `
        <span class="hidden sm:inline text-sm text-gray-600">Hi, ${hlEscapeHtml(session.name)}</span>
        <button onclick="hlLogout()" class="bg-gray-900 hover:bg-black text-white text-sm font-semibold px-4 py-2 rounded-full transition cursor-pointer">
            Log Out
        </button>
    `;
}

document.addEventListener('DOMContentLoaded', hlRenderNavAuthState);
