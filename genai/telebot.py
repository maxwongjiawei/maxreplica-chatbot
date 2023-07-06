import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from modelutils import initiate_all
import pickle
import pandas as pd
import datetime

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

PROMPT = 'chatbot_prompt'

qa = initiate_all(PROMPT)
token = '6234433703:AAEJOljbZS-lU2FqPw7FCKat9ScG-qKKuwo'
chat_history = []
all_chat_history = pd.DataFrame(columns=['query', 'answer', 'reaction', 'comments'])
filename = datetime.datetime.now().strftime('%Y%m%d %H%M')

async def handle_reaction(update, context):
    query = update.callback_query
    print(query.data)
    await query.answer("Thanks for your reaction!")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(update.message.text)
    query = update.message.text
    global chat_history
    global all_chat_history

    if query[0] == '👍':
        all_chat_history.loc[all_chat_history.index[-1], 'reaction'] = 'good'
        all_chat_history.loc[all_chat_history.index[-1], 'comments'] = query[1:]
        all_chat_history.to_csv(filename + '.csv', index=False)
        await context.bot.send_message(chat_id=update.effective_chat.id, text='thanks')
    elif query[0] == '👎':
        all_chat_history.loc[all_chat_history.index[-1], 'reaction'] = 'bad'
        all_chat_history.loc[all_chat_history.index[-1], 'comments'] = query[1:]
        all_chat_history.to_csv(filename + '.csv', index=False)
        await context.bot.send_message(chat_id=update.effective_chat.id, text='thanks')
    else:
        reply = qa({"question": query, 'chat_history': chat_history})['answer']
        chat_history = chat_history + [(query, reply)]
        all_chat_history = pd.concat([all_chat_history, pd.DataFrame({'query': [query],
                                                                      'answer': [reply],
                                                                      'reaction': [None],
                                                                      'comments': [None]})], ignore_index=True)

        all_chat_history.to_csv(filename + '.csv', index=False)
        print(all_chat_history.head())
        if len(chat_history) > 10:
            chat_history.pop(0)
        print(reply)
        print(chat_history)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)


if __name__ == '__main__':
    application = ApplicationBuilder().token(token).build()

    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    application.run_polling()
