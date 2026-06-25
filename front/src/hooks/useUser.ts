import { useState, useCallback, useEffect } from 'react';
import { createUser, getUser } from '../lib/api';
import type { UserInfo, UserState } from '../types';

const USER_STORAGE_KEY = 'hospital_guide_user';

export function useUser() {
  const [user, setUser] = useState<UserState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(USER_STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setUser(parsed);
      } catch {
        setShowModal(true);
      }
    } else {
      setShowModal(true);
    }
  }, []);

  const saveUser = useCallback((userState: UserState) => {
    localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(userState));
    setUser(userState);
    setShowModal(false);
  }, []);

  const login = useCallback(async (userId: string, name?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await createUser(userId, name);
      saveUser({ user_id: userId, name });
    } catch (err) {
      const storedUser = { user_id: userId, name };
      saveUser(storedUser);
    } finally {
      setIsLoading(false);
    }
  }, [saveUser]);

  const logout = useCallback(() => {
    localStorage.removeItem(USER_STORAGE_KEY);
    setUser(null);
    setShowModal(true);
  }, []);

  const openModal = useCallback(() => setShowModal(true), []);
  const closeModal = useCallback(() => setShowModal(false), []);

  return {
    user,
    isLoading,
    error,
    showModal,
    login,
    logout,
    openModal,
    closeModal,
  };
}
