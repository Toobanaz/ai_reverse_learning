import { useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';
import BodyLanguageMonitor, { BodyMetrics } from './BodyLanguageMonitor';
import { AudienceLevel, Message, Mode } from '../types';
import { analyzeContent } from '../utils/apiService';

interface ChatInterfaceProps {
  audienceLevel: AudienceLevel;
  mode: Mode;
  sessionId?: string;
}

const buildIntroMessage = (audienceLevel: AudienceLevel): Message => ({
  id: uuidv4(),
  content: `I'm your AI audience at the ${audienceLevel} level. I'll listen, ask questions, and give feedback on your explanations. What would you like to teach me today?`,
  sender: 'ai',
  timestamp: new Date(),
  feedback: {
    id: uuidv4(),
    summary: '',
    type: 'neutral',
    clarity: 'Ready to learn!',
    pacing: "Let's go at your pace",
    structureSuggestions: [],
    deliveryTips: [],
    questions: ['What topic would you like to explain?'],
  },
});

const ChatInterface = ({ audienceLevel, mode, sessionId }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([buildIntroMessage(audienceLevel)]);
  const [expandedMessageIds, setExpandedMessageIds] = useState<string[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [bodyMetrics, setBodyMetrics] = useState<BodyMetrics | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages([buildIntroMessage(audienceLevel)]);
    setIsTyping(false);
    setExpandedMessageIds([]);
  }, [audienceLevel, mode, sessionId]);

  useEffect(() => {
    if (mode !== 'Presentation') {
      setBodyMetrics(null);
    }
  }, [mode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleMessageExpand = (messageId: string) => {
    setExpandedMessageIds(prev =>
      prev.includes(messageId)
        ? prev.filter(id => id !== messageId)
        : [...prev, messageId]
    );
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: uuidv4(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      const isSummarize = mode === 'Explain' && content.trim().toLowerCase().startsWith('summarize');
      const transcriptSoFar = isSummarize
        ? messages.filter(m => m.sender === 'user').map(m => m.content).join('\n\n')
        : undefined;

      if (!sessionId) {
        console.warn('Missing sessionId when sending message.');
        setIsTyping(false);
        return;
      }

      const aiResponse = await analyzeContent({
        audienceLevel,
        mode,
        sessionId,
        summarize: isSummarize,
        transcriptSoFar,
        message: content,
      });

      setTimeout(() => {
        const aiMessage: Message = {
          id: uuidv4(),
          content: aiResponse.message,
          sender: 'ai',
          timestamp: new Date(),
          feedback: aiResponse.feedback
            ? {
                id: uuidv4(),
                summary: aiResponse.feedback.summary || '',
                type: aiResponse.feedback.type || 'neutral',
                clarity: aiResponse.feedback.clarity || '',
                pacing: aiResponse.feedback.pacing || '',
                structureSuggestions: aiResponse.feedback.structureSuggestions || [],
                deliveryTips: aiResponse.feedback.deliveryTips || [],
                rephrasingSuggestions: aiResponse.feedback.rephrasingSuggestions || [],
                questions: aiResponse.feedback.questions || [],
              }
            : undefined,
        };

        setMessages(prev => [...prev, aiMessage]);
        setExpandedMessageIds(prev => [...prev, aiMessage.id]);
        setIsTyping(false);
      }, 500);
    } catch (error) {
      console.error(error);
      setIsTyping(false);
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          content: "Sorry, I'm having trouble processing your request right now. Please try again later.",
          sender: 'ai',
          timestamp: new Date(),
        },
      ]);
    }
  };

  return (
    <div className="flex h-full bg-white overflow-hidden">
      {mode === 'Presentation' && (
        <div className="w-80 min-w-[280px] border-r flex flex-col overflow-hidden">
          <h4 className="p-2 text-sm font-medium">Body Language Monitor</h4>
          <div className="px-2">
            <BodyLanguageMonitor
              active={mode === 'Presentation'}
              onMetricsChange={setBodyMetrics}
            />
          </div>

          <div className="flex-grow overflow-y-auto p-2">
            <h5 className="text-sm font-medium mb-2">Analytics & Suggestions</h5>
            {bodyMetrics ? (
              <ul className="text-sm space-y-1">
                <li>Posture: {bodyMetrics.postureScore}% upright</li>
                <li>Hand gestures: {bodyMetrics.handGestureRate} /min</li>
                <li>Head nods: {bodyMetrics.headNodCount} /min</li>
                {bodyMetrics.suggestions.map((suggestion, index) => (
                  <li key={index}>Tip: {suggestion}</li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-gray-500">Enable camera access to start body analysis.</p>
            )}
          </div>
        </div>
      )}

      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 pt-16 lg:pt-4">
          {messages.map(message => (
            <MessageBubble
              key={message.id}
              message={message}
              mode={mode}
              expanded={expandedMessageIds.includes(message.id)}
              onToggleExpand={() => toggleMessageExpand(message.id)}
            />
          ))}

          {isTyping && (
            <div className="flex items-center ml-4 mb-4">
              <div className="bg-ailearn-lightgray rounded-lg px-4 py-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <MessageInput
          onSendMessage={handleSendMessage}
          audienceLevel={audienceLevel}
          mode={mode}
        />
      </div>
    </div>
  );
};

export default ChatInterface;
export { ChatInterface };
