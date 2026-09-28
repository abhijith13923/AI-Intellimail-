import { useEffect, useState, useCallback } from 'react';
import { BarChart2, MailOpen, MailCheck, Inbox, RefreshCw, Users } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * A small animated number that counts up from 0 to `target` when it appears.
 */
function AnimatedNumber({ target }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (target === 0) { setDisplay(0); return; }
    let start = 0;
    const duration = 700; // ms
    const steps = 30;
    const increment = target / steps;
    const interval = setInterval(() => {
      start += increment;
      if (start >= target) {
        setDisplay(target);
        clearInterval(interval);
      } else {
        setDisplay(Math.floor(start));
      }
    }, duration / steps);
    return () => clearInterval(interval);
  }, [target]);

  return <span>{display.toLocaleString()}</span>;
}

export default function InboxAnalytics({ userEmail, refreshTrigger }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAnalytics = useCallback(async () => {
    if (!userEmail) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `http://localhost:8000/api/analytics?user_email=${encodeURIComponent(userEmail)}`
      );
      if (!res.ok) throw new Error('Failed to fetch analytics');
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error('Analytics fetch error:', err);
      setError('Could not load stats.');
    } finally {
      setLoading(false);
    }
  }, [userEmail]);

  // Fetch on mount and whenever a sync completes (refreshTrigger changes)
  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics, refreshTrigger]);

  const readPct = data && data.total_indexed > 0
    ? Math.round((data.read_count / data.total_indexed) * 100)
    : 0;

  return (
    <div className="analytics-panel">
      {/* Header */}
      <div className="analytics-header">
        <div className="analytics-header-left">
          <BarChart2 className="analytics-icon" />
          <span>Inbox Analytics</span>
        </div>
        <button
          onClick={fetchAnalytics}
          disabled={loading}
          className="analytics-refresh-btn"
          title="Refresh stats"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <AnimatePresence mode="wait">
        {loading && !data ? (
          <motion.div
            key="skeleton"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="analytics-skeleton-wrapper"
          >
            {[1, 2, 3].map(i => (
              <div key={i} className="analytics-skeleton-card" style={{ animationDelay: `${i * 0.1}s` }} />
            ))}
          </motion.div>
        ) : error ? (
          <motion.p key="error" className="analytics-error">{error}</motion.p>
        ) : data && !data.has_data ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="analytics-empty"
          >
            <Inbox className="analytics-empty-icon" />
            <p>Sync your emails to see inbox stats.</p>
          </motion.div>
        ) : data ? (
          <motion.div
            key="stats"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Stat cards row */}
            <div className="analytics-cards">
              <div className="analytics-card analytics-card--total">
                <span className="analytics-card-label">Indexed</span>
                <span className="analytics-card-value">
                  <AnimatedNumber target={data.total_indexed} />
                </span>
              </div>
              <div className="analytics-card analytics-card--unread">
                <MailOpen className="analytics-card-icon" />
                <span className="analytics-card-label">Unread</span>
                <span className="analytics-card-value analytics-card-value--unread">
                  <AnimatedNumber target={data.unread_count} />
                </span>
              </div>
              <div className="analytics-card analytics-card--read">
                <MailCheck className="analytics-card-icon" />
                <span className="analytics-card-label">Read</span>
                <span className="analytics-card-value analytics-card-value--read">
                  <AnimatedNumber target={data.read_count} />
                </span>
              </div>
            </div>

            {/* Read/Unread progress bar */}
            {data.total_indexed > 0 && (
              <div className="analytics-progress-wrapper">
                <div className="analytics-progress-labels">
                  <span className="analytics-progress-label-read">Read {readPct}%</span>
                  <span className="analytics-progress-label-unread">Unread {100 - readPct}%</span>
                </div>
                <div className="analytics-progress-track">
                  <motion.div
                    className="analytics-progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${readPct}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  />
                </div>
              </div>
            )}

            {/* Top Senders */}
            {data.top_senders && data.top_senders.length > 0 && (
              <div className="analytics-senders">
                <div className="analytics-senders-header">
                  <Users className="w-3 h-3" />
                  <span>Top Senders</span>
                </div>
                <div className="analytics-senders-list">
                  {data.top_senders.map((sender, idx) => {
                    const maxCount = data.top_senders[0].count;
                    const barPct = Math.round((sender.count / maxCount) * 100);
                    return (
                      <div key={idx} className="analytics-sender-row">
                        <span className="analytics-sender-name" title={sender.name}>
                          {sender.name}
                        </span>
                        <div className="analytics-sender-bar-track">
                          <motion.div
                            className="analytics-sender-bar-fill"
                            initial={{ width: 0 }}
                            animate={{ width: `${barPct}%` }}
                            transition={{ duration: 0.6, delay: idx * 0.08, ease: 'easeOut' }}
                          />
                        </div>
                        <span className="analytics-sender-count">{sender.count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
